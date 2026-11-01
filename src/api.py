"""Bhumi Vikas Sahayak — FastAPI + Gemini RAG + Firebase Authentication/Firestore.

Run from project root: python -m uvicorn src.api:app --reload

Requires .env with FIREBASE_WEB_API_KEY and GOOGLE_APPLICATION_CREDENTIALS.
GEMINI_API_KEY may be set in .env or in the shell launching Uvicorn.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

import firebase_admin
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from firebase_admin import auth, firestore
from google.genai import errors as gemini_errors
from pydantic import BaseModel, EmailStr, Field

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
load_dotenv(ROOT / ".env")

logger = logging.getLogger("bhumi_vikas_api")
app = FastAPI(title="Bhumi Vikas Sahayak", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

MAX_CHATS = 10
MAX_MESSAGES = 60
MAX_HISTORY_BYTES = 700_000
SESSION_DAYS = 5
COOKIE_NAME = "bvs_auth"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

# In-memory rate limiting is for local/single-process development.
_limit_data = defaultdict(list)
_limit_lock = Lock()


def rate_limit(key: str, count: int, seconds: int) -> None:
    now = time.monotonic()
    with _limit_lock:
        timestamps = [t for t in _limit_data[key] if now - t < seconds]
        if len(timestamps) >= count:
            raise HTTPException(429, "Too many requests. Please try again later.")
        timestamps.append(now)
        _limit_data[key] = timestamps
        # Trim expired entries occasionally to bound memory use.
        if len(_limit_data) > 3000:
            for item in list(_limit_data):
                if not _limit_data[item] or now - _limit_data[item][-1] > 3600:
                    del _limit_data[item]


def client_key(request: Request) -> str:
    # Do not trust X-Forwarded-For directly from arbitrary clients.
    return request.client.host if request.client else "unknown"


def check_origin(request: Request) -> None:
    """Allow requests only from explicitly trusted frontend origins."""

    origin = request.headers.get("origin")

    # Requests without an Origin header are allowed
    # for command-line/server-side testing.
    if not origin:
        return

    allowed_origins = {
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    }

    # Additional origins for mobile testing or port forwarding.
    extra_origins = os.getenv("ALLOWED_ORIGINS", "")

    allowed_origins.update(
        item.strip().rstrip("/")
        for item in extra_origins.split(",")
        if item.strip()
    )

    if origin.rstrip("/") not in allowed_origins:
        raise HTTPException(
            status_code=403,
            detail="Request origin not allowed."
        )


@lru_cache(maxsize=1)
def db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


@lru_cache(maxsize=1)
def generator():
    from .generator import RuleGenerator
    return RuleGenerator()


def firebase_auth_call(endpoint: str, body: dict) -> dict:
    """Call Firebase Identity Toolkit from Python, not the browser."""
    api_key = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "Firebase login is not configured on the server.")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}"
    try:
        response = httpx.post(
            url, params={"key": api_key}, json=body,
            timeout=15.0,
        )
    except httpx.RequestError as exc:
        logger.warning("Firebase Auth network request failed: %s", type(exc).__name__)
        raise HTTPException(503, "Login service is temporarily unavailable.") from exc
    if response.status_code >= 400:
        try:
            error_code = response.json().get("error", {}).get("message", "")
        except ValueError:
            error_code = ""
        if error_code in ("OPERATION_NOT_ALLOWED", "CONFIGURATION_NOT_FOUND"):
            raise HTTPException(503, "Enable Email/Password in Firebase Authentication.")
        if endpoint == "signUp":
            raise HTTPException(400, "Unable to create account. Check the email and password, or try signing in.")
        if endpoint == "sendOobCode":
            # Don't reveal whether the email address exists.
            return {}
        if response.status_code == 429:
            raise HTTPException(429, "Too many login attempts. Try again later.")
        raise HTTPException(401, "Incorrect email or password.")
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(502, "Unexpected response from Firebase Authentication.") from exc


def establish_session(id_token: str, response: Response) -> dict:
    """Issue an HttpOnly, SameSite Firebase session cookie."""
    db()  # Ensure the Firebase Admin app is initialized.
    try:
        claims = auth.verify_id_token(id_token)
        if time.time() - claims.get("auth_time", 0) > 300:
            raise HTTPException(401, "Please sign in again.")
        cookie = auth.create_session_cookie(
            id_token, expires_in=timedelta(days=SESSION_DAYS)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not establish Firebase session")
        raise HTTPException(503, "Could not start a login session.") from exc

    response.set_cookie(
        key=COOKIE_NAME,
        value=cookie,
        max_age=SESSION_DAYS * 24 * 3600,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    profile = auth.get_user(claims["uid"])
    return {
        "authenticated": True,
        "uid": claims["uid"],
        "email": claims.get("email", ""),
        "name": profile.display_name or "",
    }


def current_user(request: Request) -> dict:
    """Verify the login session on EVERY protected API request."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        raise HTTPException(401, "Please sign in to continue.")
    db()
    try:
        claims = auth.verify_session_cookie(cookie, check_revoked=True)
        return {"uid": claims["uid"], "email": claims.get("email", "")}
    except Exception as exc:
        # Never expose cookie tokens or Firebase internals in responses.
        logger.info("Invalid or expired login session: %s", type(exc).__name__)
        raise HTTPException(401, "Your session has expired. Please sign in again.") from exc


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class RegistrationCredentials(Credentials):
    full_name: str = Field(min_length=2, max_length=80)


class ProfileUpdate(BaseModel):
    full_name: str = Field(min_length=2, max_length=80)


class PasswordReset(BaseModel):
    email: EmailStr


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class ChatCitation(BaseModel):
    rule: str = Field(default="", max_length=50)
    title: str = Field(default="", max_length=200)
    quote: str = Field(default="", max_length=1000)
    page_start: int | None = None
    page_end: int | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=5000)
    status: Literal["answered", "not_found", "needs_verification"] | None = None
    citations: list[ChatCitation] = Field(default_factory=list, max_length=5)


class SaveChatRequest(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    title: str = Field(default="", max_length=80)
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/source/english.pdf")
def english_source():
    path = ROOT / "data" / "bvn06032025E.pdf"
    if not path.is_file():
        raise HTTPException(404, "Source PDF unavailable.")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/health")
def health():
    return {"status": "ok", "source_version": "2025-03-06"}


# --------------------------- Login ---------------------------

@app.post("/api/auth/register", status_code=201)
def register(data: RegistrationCredentials, response: Response, request: Request):
    check_origin(request)
    rate_limit(f"register:{client_key(request)}", 5, 3600)

    # Whitespace normalization prevents awkward avatars/profile labels.
    full_name = " ".join(data.full_name.split())
    if len(full_name) < 2 or not any(character.isalpha() for character in full_name):
        raise HTTPException(422, "Enter a valid full name.")

    payload = firebase_auth_call("signUp", {
        "email": str(data.email),
        "password": data.password,
        "returnSecureToken": True,
    })

    # Store the name on the Firebase Auth user, not in browser storage.
    # Firestore conversations continue to belong to the same Firebase UID.
    db()  # Initialize Firebase Admin SDK.
    try:
        auth.update_user(payload["localId"], display_name=full_name)
    except Exception as exc:
        logger.exception("Failed to save new account display name")
        raise HTTPException(
            503,
            "The account was created, but its display name could not be saved. "
            "Please sign in and contact the administrator."
        ) from exc

    return establish_session(payload["idToken"], response)


@app.post("/api/auth/login")
def login(data: Credentials, response: Response, request: Request):
    check_origin(request)
    rate_limit(f"login:{client_key(request)}", 12, 900)
    payload = firebase_auth_call("signInWithPassword", {
        "email": str(data.email),
        "password": data.password,
        "returnSecureToken": True,
    })
    return establish_session(payload["idToken"], response)


@app.post("/api/auth/logout")
def logout(response: Response, request: Request):
    check_origin(request)
    response.delete_cookie(COOKIE_NAME, path="/", samesite="strict", secure=COOKIE_SECURE)
    return {"authenticated": False}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    # Look up the current Auth profile so a refresh preserves the name.
    profile = auth.get_user(user["uid"])
    return {"authenticated": True, **user, "name": profile.display_name or ""}


@app.patch("/api/auth/profile")
def update_profile(
    data: ProfileUpdate,
    request: Request,
    user: dict = Depends(current_user),
):
    """Allow existing accounts to set or change their display name."""
    check_origin(request)
    rate_limit(f"profile:{user['uid']}", 10, 3600)
    full_name = " ".join(data.full_name.split())
    if len(full_name) < 2 or not any(c.isalpha() for c in full_name):
        raise HTTPException(422, "Enter a valid full name.")
    try:
        auth.update_user(user["uid"], display_name=full_name)
    except Exception as exc:
        logger.exception("Failed to update Firebase display name")
        raise HTTPException(503, "Could not update your name. Try again later.") from exc
    return {"authenticated": True, **user, "name": full_name}


@app.post("/api/auth/password-reset")
def password_reset(data: PasswordReset, request: Request):
    check_origin(request)
    rate_limit(f"password-reset:{client_key(request)}", 3, 900)
    firebase_auth_call("sendOobCode", {
        "requestType": "PASSWORD_RESET", "email": str(data.email)
    })
    return {"message": "If the account exists, a password-reset email will be sent."}


# ---------------------- RAG endpoint ------------------------

@app.post("/api/ask")
def ask(data: QuestionRequest, request: Request, user: dict = Depends(current_user)):
    check_origin(request)
    question = data.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    rate_limit(f"ask:{user['uid']}", 8, 60)

    # Read history from Firestore using the authenticated user's UID.
    # The browser only supplies a chat ID, never authoritative history.
    history = []
    if data.chat_id is not None:
        chat = next(
            (c for c in read_chats(user["uid"]) if c["id"] == data.chat_id),
            None,
        )
        if chat is None:
            raise HTTPException(404, "Conversation not found.")
        history = chat.get("messages", [])[-6:]

    try:
        result = generator().generate(question, top_k=5, history=history)
    except gemini_errors.ClientError as exc:
        if exc.code == 429:
            raise HTTPException(429, "Gemini usage limit reached. Try again later.") from exc
        logger.warning("Gemini client error %s", exc.code)
        raise HTTPException(502, "Gemini could not process this question.") from exc
    except gemini_errors.ServerError as exc:
        logger.warning("Gemini server error %s", exc.code)
        raise HTTPException(503, "Gemini is temporarily unavailable.") from exc
    except Exception as exc:
        logger.exception("RAG generation failed")
        raise HTTPException(500, "The RAG backend encountered an error.") from exc
    return {
        "status": result.get("status", "not_found"),
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "model_used": result.get("model_used"),
        "source_version": "2025-03-06",
    }


# ---------------------- Chat history ------------------------

def history_reference(uid: str):
    # Firebase Auth UID is the document ID — different users never share chats.
    return db().collection("chat_history").document(uid)


def read_chats(uid: str):
    snap = history_reference(uid).get()
    return snap.to_dict().get("chats", []) if snap.exists else []


def check_history_size(chats: list) -> None:
    if len(json.dumps(chats, ensure_ascii=False).encode("utf-8")) > MAX_HISTORY_BYTES:
        raise HTTPException(413, "Chat history is too large; delete an older chat.")


@app.get("/api/chats")
def list_chats(user: dict = Depends(current_user)):
    chats = sorted(read_chats(user["uid"]), key=lambda c: c.get("updatedAt", 0), reverse=True)
    return [{
        "id": c["id"], "title": c["title"],
        "createdAt": c["createdAt"], "updatedAt": c["updatedAt"],
        "messageCount": len(c.get("messages", [])),
    } for c in chats[:MAX_CHATS]]


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user: dict = Depends(current_user)):
    for chat in read_chats(user["uid"]):
        if chat["id"] == chat_id:
            return chat
    raise HTTPException(404, "Conversation not found.")


@app.post("/api/chats")
def save_chat(data: SaveChatRequest, request: Request, user: dict = Depends(current_user)):
    check_origin(request)
    ref = history_reference(user["uid"])
    chat_id = data.id or uuid.uuid4().hex
    messages = [m.model_dump() for m in data.messages]

    @firestore.transactional
    def txn_save(transaction):
        snapshot = ref.get(transaction=transaction)
        chats = snapshot.to_dict().get("chats", []) if snapshot.exists else []
        previous = next((c for c in chats if c["id"] == chat_id), None)
        if data.id and previous is None:
            raise HTTPException(404, "Conversation no longer exists.")
        now = time.time_ns()
        title = (data.title.strip() or (previous["title"] if previous else "")
                 or messages[0]["text"][:65])[:80]
        current = {
            "id": chat_id, "title": title, "messages": messages,
            "createdAt": previous["createdAt"] if previous else now,
            "updatedAt": now,
        }
        updated = [c for c in chats if c["id"] != chat_id] + [current]
        updated.sort(key=lambda c: c["updatedAt"], reverse=True)
        updated = updated[:MAX_CHATS]
        check_history_size(updated)
        transaction.set(ref, {"chats": updated, "updatedAt": firestore.SERVER_TIMESTAMP})
        return {"id": chat_id, "title": title, "saved": True}

    return txn_save(db().transaction())


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, request: Request, user: dict = Depends(current_user)):
    check_origin(request)
    ref = history_reference(user["uid"])

    @firestore.transactional
    def txn_delete(transaction):
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise HTTPException(404, "Conversation not found.")
        chats = snapshot.to_dict().get("chats", [])
        updated = [c for c in chats if c["id"] != chat_id]
        if len(updated) == len(chats):
            raise HTTPException(404, "Conversation not found.")
        transaction.set(ref, {"chats": updated, "updatedAt": firestore.SERVER_TIMESTAMP})
        return {"deleted": True, "id": chat_id}

    return txn_delete(db().transaction())
