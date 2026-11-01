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
  authNameField: $('auth-name-field'), authName: $('auth-name'),
  authEmail: $('auth-email'), authPassword: $('auth-password'),
  authError: $('auth-error'), authSubmit: $('auth-submit'),
  authTitle: $('auth-title'), authDescription: $('auth-description'),
  authSwitch: $('auth-switch'), authSwitchLabel: $('auth-switch-label'),
  authReset: $('auth-reset'), signOut: $('sign-out'), accountEmail: $('account-email'),
  editProfile: $('edit-profile'), profileDialog: $('profile-dialog'),
  profileForm: $('profile-form'), profileName: $('profile-name'),
  profileError: $('profile-error'), profileCancel: $('profile-cancel'),
  profileSave: $('profile-save')
};

let currentChatId = null;
let messages = [];
let busy = false;
let authenticated = false;
let authMode = 'login';
let userDisplayName = '';
let userEmail = '';
let userInitials = 'U';

function initialsFromName(name, email = '') {
  const words = String(name || '').trim().split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    return (Array.from(words[0])[0] + Array.from(words[words.length - 1])[0]).toLocaleUpperCase();
  }
  if (words.length === 1) return Array.from(words[0])[0].toLocaleUpperCase();
  const localPart = String(email).split('@')[0];
  return Array.from(localPart)[0]?.toLocaleUpperCase() || 'U';
}

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
    throw new Error('Cannot connect to the server.');
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

// Render Markdown inline formatting using safe DOM nodes, never raw innerHTML.
// Supports **bold**, __bold__, *italic*, _italic_ and `inline code`.
function appendInlineFormatting(parent, text) {
  const pattern = /(\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`|\*([^*\n]+)\*|_([^_\n]+)_)/g;
  let lastIndex = 0;

  for (const match of text.matchAll(pattern)) {
    const index = match.index;
    if (index > lastIndex) {
      parent.append(document.createTextNode(text.slice(lastIndex, index)));
    }

    if (match[2] !== undefined || match[3] !== undefined) {
      parent.append(el('strong', '', match[2] ?? match[3]));
    } else if (match[4] !== undefined) {
      parent.append(el('code', '', match[4]));
    } else {
      parent.append(el('em', '', match[5] ?? match[6]));
    }
    lastIndex = index + match[0].length;
  }

  parent.append(document.createTextNode(text.slice(lastIndex)));
}

function renderReadableAnswer(text) {
  const wrap = el('div', 'answer-rich');
  let activeList = null;
  let listKind = null;
  for (const line of text.replace(/\r\n/g, '\n').split('\n')) {
    if (!line.trim()) {
      activeList = null;
      listKind = null;
      wrap.append(el('div', 'answer-gap'));
      continue;
    }
    const bullet = line.match(/^\s*[-•*]\s+(.+)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const kind = numbered ? 'ol' : 'ul';
      if (kind !== listKind || !activeList) {
        activeList = document.createElement(kind);
        wrap.append(activeList);
        listKind = kind;
      }
      const li = document.createElement('li');
      appendInlineFormatting(li, (bullet || numbered)[1]);
      activeList.append(li);
      continue;
    }
    activeList = null;
    listKind = null;
    const heading = line.match(/^\s*#{1,3}\s+(.+)$/);
    const paragraph = el(heading ? 'h3' : 'p', '', '');
    appendInlineFormatting(paragraph, heading ? heading[1] : line);
    wrap.append(paragraph);
  }
  return wrap;
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
  // Do not scroll the welcome dashboard out of view on initial login.
  if (!messages.length && !busy) return;
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
  if (!messages.length) ui.chatScroll.scrollTop = 0;

  messages.forEach((message) => {
    const isUser = message.role === 'user';
    const row = el('article', `message ${isUser ? 'user' : 'assistant'}`);
    row.append(el('div', `assistant-avatar ${isUser ? 'user-avatar' : ''}`, isUser ? userInitials : 'भू'));

    const body = el('div', 'message-body');
    body.append(el('div', 'message-heading', isUser ? (userDisplayName || 'YOU').toLocaleUpperCase() : 'BHUMI VIKAS SAHAYAK'));
    // Format assistant answers as readable paragraphs/lists without parsing HTML.
    // Every fragment uses textContent: model text cannot inject markup/scripts.
    const bubble = el('div', 'message-bubble');
    if (isUser) {
      bubble.textContent = String(message.text || '');
    } else {
      bubble.append(renderReadableAnswer(String(message.text || '')));
    }
    body.append(bubble);

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
    setStatus('Saved Chat');
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
    setStatus('Saved');
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
      ? `Your answer is visible, but we could not save it: ${error.message}`
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
  userDisplayName = '';
  userEmail = '';
  userInitials = 'U';
  setBusy(false);
  setAuthError();
}

function showApp(user) {
  authenticated = true;
  userDisplayName = String(user.name || '').trim();
  userEmail = user.email || '';
  userInitials = initialsFromName(userDisplayName, userEmail);
  ui.accountEmail.textContent = userDisplayName || user.email || 'My account';
  ui.accountEmail.title = user.email || 'Signed-in account';
  ui.authScreen.hidden = true;
  ui.appShell.hidden = false;
  ui.authPassword.value = '';
  setAuthError();
  setBusy(false);
  renderMessages();
  refreshHistory().catch((error) => {
    ui.chatList.replaceChildren(el('p', 'history-placeholder', 'Saved chats are unavailable.'));
    showNotice(`Cannot load chat history: ${error.message}`);
  });
}

function setAuthMode(mode) {
  authMode = mode;
  const signup = mode === 'register';
  ui.authTitle.textContent = signup ? 'Create your account' : 'Welcome back';
  ui.authDescription.textContent = signup
    ? 'Create an account to keep up to 10 private conversations.'
    : 'Sign in to access your saved rulebook conversations.';
  ui.authSubmit.textContent = signup ? 'Create account' : 'Sign in';
  ui.authSwitchLabel.textContent = signup ? 'Already have an account?' : 'New to Bhumi Vikas Sahayak?';
  ui.authSwitch.textContent = signup ? 'Sign in' : 'Create an account';
  ui.authPassword.autocomplete = signup ? 'new-password' : 'current-password';
  ui.authNameField.hidden = !signup;
  ui.authName.required = signup;
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
  const fullName = ui.authName.value.trim().replace(/\s+/g, ' ');
  if (!email || !password) return;
  if (authMode === 'register' && fullName.length < 2) {
    setAuthError('Enter your full name.');
    return;
  }
  ui.authSubmit.disabled = true;
  ui.authSubmit.textContent = authMode === 'login' ? 'Signing in…' : 'Creating account…';
  setAuthError();
  try {
    const user = await request(`/api/auth/${authMode}`, {
      method: 'POST', body: JSON.stringify({
        email, password,
        ...(authMode === 'register' ? { full_name: fullName } : {})
      })
    });
    showApp(user);
  } catch (error) {
    setAuthError(error.message);
  } finally {
    ui.authSubmit.disabled = false;
    ui.authSubmit.textContent = authMode === 'login' ? 'Sign in' : 'Create account';
  }
});

// Existing users can add a name without creating another account.
ui.editProfile.addEventListener('click', () => {
  if (!authenticated) return;
  ui.profileName.value = userDisplayName;
  ui.profileError.hidden = true;
  ui.profileError.textContent = '';
  ui.profileDialog.showModal();
  ui.profileName.focus();
});
ui.profileCancel.addEventListener('click', () => ui.profileDialog.close());
ui.profileForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const fullName = ui.profileName.value.trim().replace(/\s+/g, ' ');
  if (fullName.length < 2) {
    ui.profileError.textContent = 'Enter your full name.';
    ui.profileError.hidden = false;
    return;
  }
  ui.profileSave.disabled = true;
  ui.profileError.hidden = true;
  try {
    const profile = await request('/api/auth/profile', {
      method: 'PATCH', body: JSON.stringify({ full_name: fullName })
    });
    userDisplayName = profile.name || '';
    userEmail = profile.email || userEmail;
    userInitials = initialsFromName(userDisplayName, userEmail);
    ui.accountEmail.textContent = userDisplayName || userEmail;
    ui.accountEmail.title = userEmail;
    renderMessages();
    ui.profileDialog.close();
  } catch (error) {
    ui.profileError.textContent = error.message;
    ui.profileError.hidden = false;
  } finally {
    ui.profileSave.disabled = false;
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


/* ------------------ Intro logo reveal + keyboard shortcuts ------------------ */
(function initializeIntro() {
  const intro = $('intro-screen');
  const skip = $('intro-skip');
  if (!intro) return;
  let seen = false;
  try { seen = sessionStorage.getItem('bvs_intro_seen') === '1'; } catch { /* storage disabled */ }
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let dismissed = false;
  if (seen || reducedMotion) {
    intro.hidden = true;
  }
  const dismiss = () => {
    if (dismissed) return;
    dismissed = true;
    intro.classList.add('is-leaving');
    try { sessionStorage.setItem('bvs_intro_seen', '1'); } catch { /* ignore */ }
    // Ensure the overlay is removed even if transitionend does not fire.
    window.setTimeout(() => { intro.hidden = true; }, 650);
  };
  skip.addEventListener('click', dismiss);
  if (!seen && !reducedMotion) window.setTimeout(dismiss, 2250);

  // Replay is useful for reviewing the logo animation during UI development.
  const replay = $('replay-intro');
  replay?.addEventListener('click', () => {
    closeSidebar();
    dismissed = false;
    intro.hidden = true;
    intro.classList.remove('is-leaving');
    // Force the element to re-enter layout to restart CSS animations.
    void intro.offsetWidth;
    intro.hidden = false;
    window.setTimeout(dismiss, 2250);
  });
})();

// Keeps the workspace pleasant to use without adding dependencies.
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    if (authenticated) {
      event.preventDefault();
      ui.input.focus();
    }
  }
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'o') {
    if (authenticated && !busy) {
      event.preventDefault();
      newConversation();
    }
  }
});

const yearTarget = $('current-year');
if (yearTarget) yearTarget.textContent = String(new Date().getFullYear());
