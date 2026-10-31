
"""
Bhumi Vikas Sahayak - FastAPI Backend

Connects:
1. Existing Gemini RAG generator
2. Firebase Firestore chat storage
3. Future HTML/CSS/JavaScript frontend

Run from project root:
    python -m uvicorn src.api:app --reload
"""

import os
import json
import time
import uuid
import logging

from pathlib import Path
from functools import lru_cache
from threading import Lock
from collections import defaultdict
from typing import Literal

import firebase_admin

from firebase_admin import firestore
from google.genai import errors as gemini_errors
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env.
# Existing PowerShell variables are NOT overwritten.
load_dotenv(ROOT / ".env")

SESSION_SECRET = os.getenv("SESSION_SECRET", "")

if len(SESSION_SECRET) < 32:
    raise RuntimeError(
        "SESSION_SECRET is missing or too short. "
        "Generate one and place it in .env."
    )

MAX_CHATS = 10
MAX_MESSAGES = 60
MAX_HISTORY_BYTES = 700_000

logger = logging.getLogger("bhumi_vikas_api")


# ==================================================
# 2. FASTAPI INITIALIZATION
# ==================================================

app = FastAPI(
    title="Bhumi Vikas Sahayak",
    description=(
        "RAG chatbot API for Madhya Pradesh "
        "Bhumi Vikas Rules, 2012."
    ),
    version="1.0.0"
)

# Anonymous session cookie for local development.
# The cookie contains only a random user identifier.
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="bvs_session",
    same_site="strict",
    https_only=(
        os.getenv("COOKIE_SECURE", "false").lower()
        == "true"
    ),
    max_age=60 * 60 * 24 * 30
)

FRONTEND_DIR = ROOT / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)

# ==================================================
# 3. FIREBASE CONNECTION
# ==================================================

@lru_cache(maxsize=1)
def get_database():
    """
    Initialize Firebase only when it is first used.

    Credentials are loaded from:
    GOOGLE_APPLICATION_CREDENTIALS
    """

    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    return firestore.client()


# ==================================================
# 4. EXISTING RAG GENERATOR
# ==================================================

@lru_cache(maxsize=1)
def get_generator():
    """
    Reuse the RuleGenerator class from generator.py.

    No changes to ingestion.py, embeddings.py,
    retriever.py or generator.py are required.
    """

    from .generator import RuleGenerator

    return RuleGenerator()


# ==================================================
# 5. BROWSER SESSION
# ==================================================

def get_user_id(request: Request):
    """
    Each browser gets a random persistent guest ID.

    This is not a full login system. It is intended
    for local development and the first prototype.
    """

    user_id = request.session.get("guest_id")

    if user_id is None:
        user_id = uuid.uuid4().hex
        request.session["guest_id"] = user_id

    return user_id


def get_chat_reference(request: Request):
    """
    One Firestore document stores at most 10 chats.

    Collection: chat_history
    Document ID: guest_id
    """

    user_id = get_user_id(request)

    return (
        get_database()
        .collection("chat_history")
        .document(user_id)
    )


# ==================================================
# 6. REQUEST MODELS
# ==================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000
    )


class ChatCitation(BaseModel):
    rule: str = Field(default="", max_length=50)
    title: str = Field(default="", max_length=200)
    quote: str = Field(default="", max_length=1000)

    page_start: int | None = None
    page_end: int | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]

    text: str = Field(
        min_length=1,
        max_length=5000
    )

    status: Literal[
        "answered",
        "not_found",
        "needs_verification"
    ] | None = None

    citations: list[ChatCitation] = Field(
        default_factory=list,
        max_length=5
    )


class SaveChatRequest(BaseModel):
    id: str | None = Field(default=None, max_length=64)

    title: str = Field(
        default="",
        max_length=80
    )

    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=MAX_MESSAGES
    )


# ==================================================
# 7. DATABASE HELPERS
# ==================================================

def get_saved_chats(request: Request):
    """Read conversations for the current session."""

    reference = get_chat_reference(request)

    snapshot = reference.get()

    if not snapshot.exists:
        return []

    return snapshot.to_dict().get("chats", [])


def check_history_size(chats):
    """
    Keep Firestore document comfortably below its
    maximum supported document size.
    """

    encoded = json.dumps(
        chats,
        ensure_ascii=False
    ).encode("utf-8")

    if len(encoded) > MAX_HISTORY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Saved conversation history is too large. "
                "Please shorten or delete an older chat."
            )
        )


# ==================================================
# 8. BASIC RATE LIMITING
# ==================================================

# In-memory limit for local development only.
# A production system needs a shared rate limiter
# and stronger user authentication.

request_times = defaultdict(list)
rate_limit_lock = Lock()


def check_rate_limit(user_id):
    """Allow up to 8 Gemini requests per minute."""

    now = time.monotonic()

    with rate_limit_lock:

        # Remove old entries for every known session.
        for uid in list(request_times):
            recent = [
                timestamp
                for timestamp in request_times[uid]
                if now - timestamp < 60
            ]

            if recent:
                request_times[uid] = recent
            else:
                del request_times[uid]

        recent = request_times[user_id]

        if len(recent) >= 8:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many questions. "
                    "Please try again in a minute."
                )
            )

        recent.append(now)


# ==================================================
# 9. HEALTH CHECK
# ==================================================

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Bhumi Vikas Sahayak API",
        "source_version": "2025-03-06"
    }


# ==================================================
# 10. ASK A QUESTION
# ==================================================

@app.post("/api/ask")
def ask_question(
    data: QuestionRequest,
    request: Request
):
    """
    Accept a question and return a grounded RAG answer.

    Uses the existing:
        RuleGenerator.generate(question, top_k=5)
    """

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    user_id = get_user_id(request)
    check_rate_limit(user_id)

    try:
        generator = get_generator()

        result = generator.generate(
            question,
            top_k=5
        )

        return {
            "status": result.get(
                "status", "not_found"
            ),
            "answer": result.get(
                "answer", ""
            ),
            "citations": result.get(
                "citations", []
            ),
            "model_used": result.get(
                "model_used"
            ),
            "source_version": "2025-03-06"
        }

    except gemini_errors.ClientError as error:

        code = error.code

        if code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota or rate limit exceeded."
            ) from error

        logger.warning(
            "Gemini client error; status=%s",
            code
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini could not process this request. "
                "Check the backend configuration."
            )
        ) from error

    except gemini_errors.ServerError as error:

        logger.warning(
            "Gemini server error; status=%s",
            error.code
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable. "
                "Please try again."
            )
        ) from error

    except Exception as error:

        logger.exception(
            "Error while generating RAG answer"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG backend encountered an error. "
                "Check the server terminal."
            )
        ) from error


# ==================================================
# 11. LIST SAVED CHATS
# ==================================================

@app.get("/api/chats")
def list_chats(request: Request):
    """Return metadata for up to 10 saved conversations."""

    chats = get_saved_chats(request)

    chats.sort(
        key=lambda chat: chat.get("updatedAt", 0),
        reverse=True
    )

    return [
        {
            "id": chat["id"],
            "title": chat["title"],
            "createdAt": chat["createdAt"],
            "updatedAt": chat["updatedAt"],
            "messageCount": len(
                chat.get("messages", [])
            )
        }
        for chat in chats[:MAX_CHATS]
    ]


# ==================================================
# 12. RETRIEVE ONE CHAT
# ==================================================

@app.get("/api/chats/{chat_id}")
def get_chat(
    chat_id: str,
    request: Request
):
    chats = get_saved_chats(request)

    for chat in chats:
        if chat["id"] == chat_id:
            return chat

    raise HTTPException(
        status_code=404,
        detail="Conversation not found."
    )


# ==================================================
# 13. CREATE OR UPDATE A CHAT
# ==================================================

@app.post("/api/chats")
def save_chat(
    data: SaveChatRequest,
    request: Request
):
    """
    Save the entire conversation.

    Existing chat ID -> update chat.
    No chat ID       -> create new chat.

    Keep only the 10 most recent conversations.
    """

    database = get_database()
    reference = get_chat_reference(request)

    chat_id = data.id or uuid.uuid4().hex

    # Pydantic has already validated the messages.
    messages = [
        message.model_dump()
        for message in data.messages
    ]

    @firestore.transactional
    def save_transaction(transaction):

        snapshot = reference.get(
            transaction=transaction
        )

        if snapshot.exists:
            chats = snapshot.to_dict().get(
                "chats", []
            )
        else:
            chats = []

        previous = next(
            (
                chat for chat in chats
                if chat["id"] == chat_id
            ),
            None
        )

        # Do not silently recreate an older,
        # deleted conversation.
        if data.id and previous is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )

        now = time.time_ns()

        title = (
            data.title.strip()
            or (
                previous["title"]
                if previous else ""
            )
            or messages[0]["text"][:65]
        )

        new_chat = {
            "id": chat_id,
            "title": title[:80],
            "messages": messages,
            "createdAt": (
                previous["createdAt"]
                if previous else now
            ),
            "updatedAt": now
        }

        # Remove old version of the conversation.
        chats = [
            chat for chat in chats
            if chat["id"] != chat_id
        ]

        # Insert latest version.
        chats.append(new_chat)

        # Newest first.
        chats.sort(
            key=lambda chat: chat["updatedAt"],
            reverse=True
        )

        # Strict 10-chat storage limit.
        chats = chats[:MAX_CHATS]

        check_history_size(chats)

        transaction.set(
            reference,
            {
                "chats": chats,
                "updatedAt": firestore.SERVER_TIMESTAMP
            }
        )

        return {
            "id": chat_id,
            "title": new_chat["title"],
            "saved": True
        }

    transaction = database.transaction()

    return save_transaction(transaction)


# ==================================================
# 14. DELETE A CHAT
# ==================================================

@app.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    request: Request
):
    database = get_database()
    reference = get_chat_reference(request)

    @firestore.transactional
    def delete_transaction(transaction):

        snapshot = reference.get(
            transaction=transaction
        )

        if not snapshot.exists:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )

        chats = snapshot.to_dict().get(
            "chats", []
        )

        updated = [
            chat for chat in chats
            if chat["id"] != chat_id
        ]

        if len(updated) == len(chats):
            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )

        transaction.set(
            reference,
            {
                "chats": updated,
                "updatedAt": firestore.SERVER_TIMESTAMP
            }
        )

        return {
            "deleted": True,
            "id": chat_id
        }

    transaction = database.transaction()

    return delete_transaction(transaction)


@app.get("/source/english.pdf")
def source_pdf():
    pdf_path = ROOT / "data" / "bvn06032025E.pdf"

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Source PDF not found."
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf"
    )