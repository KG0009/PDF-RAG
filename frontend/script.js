/* Bhumi Vikas Sahayak — Vanilla JavaScript frontend.
   Uses only FastAPI endpoints; no Gemini or Firebase keys in this file. */

const $ = (id) => document.getElementById(id);
const ui = {
  sidebar: $('sidebar'), overlay: $('sidebar-overlay'), chatList: $('chat-list'),
  chatCount: $('chat-count'), newChat: $('new-chat'), welcome: $('welcome'),
  messages: $('messages'), chatScroll: $('chat-scroll'), typing: $('typing-indicator'),
  form: $('chat-form'), input: $('question'), send: $('send-button'),
  notice: $('notice'), saveStatus: $('save-status'), themeToggle: $('theme-toggle'),
  authScreen: $('auth-screen'), appShell: $('app-shell'), authForm: $('auth-form'),
  authEmail: $('auth-email'), authPassword: $('auth-password'),
  authError: $('auth-error'), authSubmit: $('auth-submit'),
  authTitle: $('auth-title'), authDescription: $('auth-description'),
  authSwitch: $('auth-switch'), authSwitchLabel: $('auth-switch-label'),
  authReset: $('auth-reset'), signOut: $('sign-out'), accountEmail: $('account-email')
};

let currentChatId = null;
let messages = [];
let busy = false;
let authenticated = false;
let authMode = 'login';
const MAX_SAVED_MESSAGES = 60;  // Matches src/api.py

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
  } catch {
    throw new Error('Cannot connect to the server. Is FastAPI running?');
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith('/api/auth/')) {
      showLogin();
    }
    const detail = payload.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`);
  }
  return payload;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showNotice(text = '') {
  ui.notice.hidden = !text;
  ui.notice.textContent = text;
}

function setStatus(text) { ui.saveStatus.textContent = text; }

function setBusy(value) {
  busy = value;
  ui.newChat.disabled = value || !authenticated;
  ui.typing.hidden = !value;
  ui.input.disabled = value;
  updateSendButton();
  scrollBottom();
}

function updateSendButton() { ui.send.disabled = busy || !authenticated || !ui.input.value.trim(); }

function scrollBottom() {
  requestAnimationFrame(() => {
    ui.chatScroll.scrollTop = ui.chatScroll.scrollHeight;
  });
}

function cleanCitation(source) {
  const page = (value) => Number.isInteger(Number(value)) && Number(value) > 0
    ? Number(value) : null;
  return {
    rule: String(source.rule ?? '').slice(0, 50),
    title: String(source.title ?? '').slice(0, 200),
    quote: String(source.quote ?? '').slice(0, 1000),
    page_start: page(source.page_start),
    page_end: page(source.page_end)
  };
}

function normalizedMessages() {
  return messages.slice(-MAX_SAVED_MESSAGES).map((item) => ({
    role: item.role,
    text: String(item.text).slice(0, 5000),
    status: ['answered', 'not_found', 'needs_verification'].includes(item.status)
      ? item.status : null,
    citations: (Array.isArray(item.citations) ? item.citations : [])
      .slice(0, 5).map(cleanCitation)
  }));
}

function renderSources(citations) {
  const wrap = el('div', 'sources');
  wrap.append(el('div', 'sources-title', 'REFERENCED PROVISIONS'));

  citations.slice(0, 5).forEach((source, index) => {
    const card = el('details', 'source-card');
    const label = `Source ${index + 1} · Rule ${source.rule || '—'}`;
    const heading = source.title ? `${label} — ${source.title}` : label;
    card.append(el('summary', '', heading));
    const details = el('div', 'source-details');
    if (source.quote) details.append(el('blockquote', '', source.quote));

    const startPage = Number(source.page_start);
    if (Number.isInteger(startPage) && startPage >= 1) {
      const link = el('a', '', `Open source PDF near page ${startPage} ↗`);
      link.href = `/source/english.pdf#page=${startPage}`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      details.append(link);
    }
    card.append(details);
    wrap.append(card);
  });
  return wrap;
}

function renderMessages() {
  ui.welcome.hidden = messages.length > 0;
  ui.messages.replaceChildren();

  messages.forEach((message) => {
    const isUser = message.role === 'user';
    const row = el('article', `message ${isUser ? 'user' : 'assistant'}`);
    row.append(el('div', `assistant-avatar ${isUser ? 'user-avatar' : ''}`, isUser ? 'YOU' : 'भू'));

    const body = el('div', 'message-body');
    body.append(el('div', 'message-heading', isUser ? 'YOU' : 'BHUMI VIKAS SAHAYAK'));
    // textContent safely displays model output without interpreting HTML.
    body.append(el('div', 'message-bubble', String(message.text || '')));

    if (!isUser) {
      const meta = el('div', 'message-meta');
      const labels = {
        answered: 'Answer with source references',
        not_found: 'Not found in available rules',
        needs_verification: 'Needs further verification'
      };
      if (labels[message.status]) {
        meta.append(el('span', `answer-status ${message.status === 'answered' ? '' : 'warning'}`, labels[message.status]));
      }
      const copy = el('button', 'copy-button', 'Copy answer');
      copy.type = 'button';
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(String(message.text || ''));
          copy.textContent = 'Copied!';
          setTimeout(() => { copy.textContent = 'Copy answer'; }, 1300);
        } catch { showNotice('Could not copy. Please select the text instead.'); }
      });
      meta.append(copy);
      body.append(meta);
      if (Array.isArray(message.citations) && message.citations.length) {
        body.append(renderSources(message.citations));
      }
    }
    row.append(body);
    ui.messages.append(row);
  });
  scrollBottom();
}

function closeSidebar() {
  ui.sidebar.classList.remove('is-open');
  ui.overlay.hidden = true;
}
function openSidebar() {
  ui.sidebar.classList.add('is-open');
  ui.overlay.hidden = false;
}
$('open-sidebar').addEventListener('click', openSidebar);
$('close-sidebar').addEventListener('click', closeSidebar);
ui.overlay.addEventListener('click', closeSidebar);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeSidebar();
});

async function refreshHistory() {
  const chats = await request('/api/chats');
  if (!Array.isArray(chats)) throw new Error('Invalid chat history response.');
  ui.chatList.replaceChildren();
  ui.chatCount.textContent = `${chats.length} / 10`;

  if (!chats.length) {
    ui.chatList.append(el('p', 'history-placeholder', 'Your saved conversations will appear here.'));
    return;
  }

  chats.forEach((chat) => {
    const row = el('div', `chat-item ${chat.id === currentChatId ? 'active' : ''}`);
    const open = el('button', 'chat-open');
    open.type = 'button';
    open.title = chat.title || 'Untitled chat';
    open.disabled = busy;
    open.append(el('span', 'chat-title', chat.title || 'Untitled chat'));
    open.addEventListener('click', () => loadConversation(chat.id));

    const remove = el('button', 'chat-delete', '×');
    remove.type = 'button';
    remove.disabled = busy;
    remove.title = 'Delete conversation';
    remove.setAttribute('aria-label', `Delete ${chat.title || 'conversation'}`);
    remove.addEventListener('click', () => deleteConversation(chat.id));
    row.append(open, remove);
    ui.chatList.append(row);
  });
}

async function loadConversation(id) {
  if (busy || !authenticated) return;
  showNotice();
  try {
    const chat = await request(`/api/chats/${encodeURIComponent(id)}`);
    currentChatId = chat.id;
    messages = Array.isArray(chat.messages) ? chat.messages : [];
    renderMessages();
    await refreshHistory();
    setStatus('Saved in Firebase');
    closeSidebar();
  } catch (error) { showNotice(error.message); }
}

function newConversation() {
  if (busy || !authenticated) return;
  currentChatId = null;
  messages = [];
  ui.input.value = '';
  resizeInput();
  renderMessages();
  updateSendButton();
  setStatus('New conversation');
  showNotice();
  closeSidebar();
  refreshHistory().catch((error) => showNotice(error.message));
  ui.input.focus();
}
ui.newChat.addEventListener('click', newConversation);

async function deleteConversation(id) {
  if (busy || !authenticated || !window.confirm('Permanently delete this conversation?')) return;
  try {
    await request(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (id === currentChatId) {
      currentChatId = null;
      messages = [];
      renderMessages();
      setStatus('Ready');
    }
    await refreshHistory();
    showNotice();
  } catch (error) { showNotice(error.message); }
}

async function saveConversation() {
  const firstQuestion = messages.find((message) => message.role === 'user');
  const title = (firstQuestion?.text || 'New conversation').slice(0, 80);
  const payload = await request('/api/chats', {
    method: 'POST',
    body: JSON.stringify({ id: currentChatId, title, messages: normalizedMessages() })
  });
  currentChatId = payload.id;
  await refreshHistory();
}

async function sendQuestion(text) {
  const question = String(text || '').trim();
  if (!question || busy || !authenticated) return;
  showNotice();
  ui.input.value = '';
  resizeInput();
  messages.push({ role: 'user', text: question, citations: [] });
  renderMessages();
  setBusy(true);
  setStatus('Searching rules…');

  let answered = false;
  try {
    const result = await request('/api/ask', {
      method: 'POST',
      body: JSON.stringify({ question, chat_id: currentChatId })
    });
    messages.push({
      role: 'assistant',
      text: String(result.answer || 'No answer was returned.'),
      status: result.status || 'not_found',
      citations: Array.isArray(result.citations) ? result.citations : []
    });
    answered = true;
    if (messages.length > MAX_SAVED_MESSAGES) {
      messages = messages.slice(-MAX_SAVED_MESSAGES);
      showNotice('Only the latest 60 messages are retained in this conversation.');
    }
    renderMessages();
    setStatus('Saving chat…');
    await saveConversation();
    setStatus('Saved in Firebase');
  } catch (error) {
    if (!answered) {
      // Restore the unsent question so it can be retried.
      messages.pop();
      ui.input.value = question;
      resizeInput();
      renderMessages();
      setStatus('Request failed');
    } else {
      setStatus('Not saved');
    }
    showNotice(answered
      ? `Your answer is visible, but Firebase could not save it: ${error.message}`
      : error.message);
  } finally {
    setBusy(false);
    ui.input.focus();
  }
}

function resizeInput() {
  ui.input.style.height = 'auto';
  ui.input.style.height = `${Math.min(ui.input.scrollHeight, 156)}px`;
  updateSendButton();
}
ui.input.addEventListener('input', resizeInput);
ui.input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) ui.form.requestSubmit();
  }
});
ui.form.addEventListener('submit', (event) => {
  event.preventDefault();
  sendQuestion(ui.input.value);
});

document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => sendQuestion(button.dataset.question));
});

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  ui.themeToggle.textContent = theme === 'dark' ? '☀' : '☾';
  ui.themeToggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
}
let savedTheme = null;
try { savedTheme = localStorage.getItem('bvs_theme'); } catch { /* storage unavailable */ }
applyTheme(savedTheme === 'light' || savedTheme === 'dark'
  ? savedTheme : (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
ui.themeToggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('bvs_theme', next); } catch { /* storage unavailable */ }
});

/* -------------------------------- Authentication ----------------------------- */

function setAuthError(message = '') {
  ui.authError.hidden = !message;
  ui.authError.textContent = message;
}

function showLogin() {
  authenticated = false;
  ui.appShell.hidden = true;
  ui.authScreen.hidden = false;
  currentChatId = null;
  messages = [];
  ui.messages.replaceChildren();
  ui.input.value = '';
  ui.accountEmail.textContent = '';
  setBusy(false);
  setAuthError();
}

function showApp(user) {
  authenticated = true;
  ui.accountEmail.textContent = user.email || 'My account';
  ui.authScreen.hidden = true;
  ui.appShell.hidden = false;
  ui.authPassword.value = '';
  setAuthError();
  setBusy(false);
  renderMessages();
  refreshHistory().catch((error) => {
    ui.chatList.replaceChildren(el('p', 'history-placeholder', 'Saved chats are unavailable.'));
    showNotice(`Cannot load Firebase chat history: ${error.message}`);
  });
}

function setAuthMode(mode) {
  authMode = mode;
  const signup = mode === 'register';
  ui.authTitle.textContent = signup ? 'Create your account' : 'Welcome back';
  ui.authDescription.textContent = signup
    ? 'Create an account to keep up to 10 private conversations in Firebase.'
    : 'Sign in to access your saved rulebook conversations.';
  ui.authSubmit.textContent = signup ? 'Create account' : 'Sign in';
  ui.authSwitchLabel.textContent = signup ? 'Already have an account?' : 'New to Bhumi Vikas Sahayak?';
  ui.authSwitch.textContent = signup ? 'Sign in' : 'Create an account';
  ui.authPassword.autocomplete = signup ? 'new-password' : 'current-password';
  ui.authReset.hidden = signup;
  setAuthError();
}

ui.authSwitch.addEventListener('click', () => {
  setAuthMode(authMode === 'login' ? 'register' : 'login');
});

ui.authForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const email = ui.authEmail.value.trim();
  const password = ui.authPassword.value;
  if (!email || !password) return;
  ui.authSubmit.disabled = true;
  ui.authSubmit.textContent = authMode === 'login' ? 'Signing in…' : 'Creating account…';
  setAuthError();
  try {
    const user = await request(`/api/auth/${authMode}`, {
      method: 'POST', body: JSON.stringify({ email, password })
    });
    showApp(user);
  } catch (error) {
    setAuthError(error.message);
  } finally {
    ui.authSubmit.disabled = false;
    ui.authSubmit.textContent = authMode === 'login' ? 'Sign in' : 'Create account';
  }
});

ui.authReset.addEventListener('click', async () => {
  const email = ui.authEmail.value.trim();
  if (!email) {
    setAuthError('Enter your email address first, then choose Forgot password.');
    ui.authEmail.focus();
    return;
  }
  try {
    const result = await request('/api/auth/password-reset', {
      method: 'POST', body: JSON.stringify({ email })
    });
    setAuthError(result.message);
  } catch (error) { setAuthError(error.message); }
});

ui.signOut.addEventListener('click', async () => {
  if (busy) return;
  try {
    await request('/api/auth/logout', { method: 'POST' });
    showLogin();
    setAuthMode('login');
  } catch (error) { showNotice(error.message); }
});

/* Initialize the interface only after validating the Firebase session. */
applyTheme(document.documentElement.dataset.theme);
showLogin();
request('/api/auth/me').then(showApp).catch((error) => {
  // An unauthorized response is normal before the first login.
  if (!error.message.toLowerCase().includes('sign in')
      && !error.message.toLowerCase().includes('session')) {
    setAuthError(error.message);
  }
});
