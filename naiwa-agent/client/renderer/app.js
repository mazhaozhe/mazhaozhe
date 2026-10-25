const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const PROVIDERS = {
  deepseek: { label: 'DeepSeek', keyUrl: 'https://platform.deepseek.com/api_keys' },
  qwen: { label: '通义千问', keyUrl: 'https://bailian.console.aliyun.com/' },
  kimi: { label: 'Kimi', keyUrl: 'https://platform.moonshot.cn/console/api-keys' }
};

const state = {
  provider: 'deepseek',
  theme: 'day',
  hasKey: false,
  configured: false,
  sending: false,
  sessionId: null,
  sessions: [],
  models: [],
  currentModel: {},
  streamElement: null,
  journalEntries: [],
  journalId: null,
  toastTimer: null,
  attachments: [],        // 待发送附件：[{uploadId,name,size,mime,kind,isImage,preview}]
  runTimer: null,         // 任务看门狗定时器（后端超时前的双保险）
  toolTimer: null,        // 当前工具调用的计时器（让长时间命令有可见进度）
  runTimeoutSeconds: 900  // 后端单任务硬超时（health 接口下发）
};

const MAX_ATTACHMENTS = 9;

const welcomeHTML = $('#welcome').outerHTML;

function toast(message, error = false) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.toggle('error', error);
  el.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
}

function localDate() {
  const date = new Date();
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
}

function formatSize(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// 附件上传：按钮选择 / 拖拽 / 粘贴 → upload_file 暂存 → chips → 随消息发送
// ---------------------------------------------------------------------------
function currentModelSupportsVision() {
  return Boolean(state.currentModel && state.currentModel.supports_vision);
}

function renderAttachments() {
  const tray = $('#attachmentTray');
  if (!tray) return;
  tray.replaceChildren();
  if (!state.attachments.length) {
    tray.classList.add('hidden');
    return;
  }
  tray.classList.remove('hidden');
  for (const item of state.attachments) {
    const chip = document.createElement('div');
    chip.className = 'attachment-chip';
    chip.dataset.uploadId = item.uploadId;
    if (item.isImage && item.preview) {
      const thumb = document.createElement('img');
      thumb.className = 'chip-thumb';
      thumb.src = item.preview;
      thumb.alt = item.name;
      chip.append(thumb);
    } else {
      const icon = document.createElement('span');
      icon.className = 'chip-icon';
      icon.textContent = item.isImage ? '图' : '档';
      chip.append(icon);
    }
    const text = document.createElement('div');
    text.className = 'chip-text';
    const name = document.createElement('span');
    name.className = 'chip-name';
    name.textContent = item.name;
    name.title = item.name;
    const size = document.createElement('span');
    size.className = 'chip-size';
    size.textContent = formatSize(item.size);
    text.append(name, size);
    const remove = document.createElement('button');
    remove.className = 'chip-remove';
    remove.type = 'button';
    remove.textContent = '×';
    remove.title = '移除附件';
    remove.addEventListener('click', () => removeAttachment(item.uploadId));
    chip.append(text, remove);
    tray.append(chip);
  }
  // 有照片但当前模型不支持视觉：明确告知发送后只会得到"无法识别"提示
  if (state.attachments.some((item) => item.isImage) && !currentModelSupportsVision()) {
    const warning = document.createElement('div');
    warning.className = 'attachment-warning';
    warning.textContent = '当前模型不支持图片识别，发送后将提示"模型无法识别该照片"';
    tray.append(warning);
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('读取文件内容失败'));
    reader.readAsDataURL(file);
  });
}

async function attachFiles(files) {
  const incoming = [...(files || [])].filter(Boolean);
  if (!incoming.length) return;
  if (state.attachments.length + incoming.length > MAX_ATTACHMENTS) {
    toast(`一次最多上传 ${MAX_ATTACHMENTS} 个附件`, true);
    return;
  }
  for (const file of incoming) {
    try {
      let preview = '';
      let params;
      let path = '';
      try { path = window.naiwa.getPathForFile(file) || ''; } catch { path = ''; }
      if (path) {
        params = { path, name: file.name, mime: file.type || '' };
        if ((file.type || '').startsWith('image/')) preview = await readFileAsDataURL(file);
      } else {
        // 拿不到本机路径（如剪贴板构造的文件）：base64 兜底
        const dataUrl = await readFileAsDataURL(file);
        if ((file.type || '').startsWith('image/')) preview = dataUrl;
        params = { dataBase64: dataUrl.split(',')[1] || '', name: file.name, mime: file.type || '' };
      }
      const upload = await window.naiwa.request('upload_file', params);
      state.attachments.push({ ...upload, preview });
      renderAttachments();
    } catch (error) {
      toast(`上传失败：${file.name}（${error.message}）`, true);
    }
  }
}

async function removeAttachment(uploadId) {
  state.attachments = state.attachments.filter((item) => item.uploadId !== uploadId);
  renderAttachments();
  try { await window.naiwa.request('discard_upload', { uploadId }); } catch {}
}

async function clearAttachments() {
  const pending = state.attachments.map((item) => item.uploadId);
  state.attachments = [];
  renderAttachments();
  for (const uploadId of pending) {
    try { await window.naiwa.request('discard_upload', { uploadId }); } catch {}
  }
}

function attachmentChipsElement(attachments) {
  const wrap = document.createElement('div');
  wrap.className = 'message-attachments';
  for (const file of attachments || []) {
    const chip = document.createElement('span');
    chip.className = 'msg-attach-chip';
    chip.textContent = `${file.isImage ? '图片' : '文件'} ${file.name}（${formatSize(file.size || 0)}）`;
    chip.title = file.path ? '点击打开本地文件' : file.name;
    if (file.path) {
      chip.addEventListener('click', () => {
        window.naiwa.openPath(file.path).catch(() => toast('打开文件失败', true));
      });
    }
    wrap.append(chip);
  }
  return wrap;
}

function formatTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function formatDay(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  if (date.toDateString() === new Date().toDateString()) return formatTime(value);
  return date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' });
}

function switchView(name) {
  $$('.view').forEach((view) => view.classList.toggle('active', view.id === `${name}View`));
  $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === name));
  if (name === 'journal') refreshJournal();
  if (name === 'settings') renderModels();
}

function applyTheme(theme) {
  state.theme = theme === 'night' ? 'night' : 'day';
  document.documentElement.dataset.theme = state.theme;
  $('.theme-icon').textContent = state.theme === 'day' ? '☀' : '☾';
  $('.theme-label').textContent = state.theme === 'day' ? '白天模式' : '夜间模式';
  $$('.theme-option').forEach((item) => item.classList.toggle('active', item.dataset.themeChoice === state.theme));
  window.naiwa.saveSettings({ theme: state.theme }).catch(() => {});
}

function updateProvider(provider) {
  state.provider = provider;
  $$('.provider-tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.provider === provider));
  $('#apiKeyLabel').textContent = `${PROVIDERS[provider].label} API Key`;
  $('#configStatus').classList.add('hidden');
  window.naiwa.saveSettings({ provider }).catch(() => {});
  loadModels(provider);
}

function renderHistory() {
  const list = $('#historyList');
  list.replaceChildren();
  $('#historyCount').textContent = state.sessions.length;
  if (!state.sessions.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.innerHTML = '还没有对话<br>从一个想法开始吧';
    list.append(empty);
    return;
  }
  for (const session of state.sessions) {
    const item = document.createElement('div');
    item.className = `history-item${session.id === state.sessionId ? ' active' : ''}`;
    const select = document.createElement('button');
    select.className = 'history-select';
    const title = document.createElement('strong');
    title.textContent = session.title || '新对话';
    const meta = document.createElement('small');
    meta.textContent = `${formatDay(session.updated_at)} · ${session.message_count || 0} 条消息`;
    select.append(title, meta);
    select.addEventListener('click', () => selectSession(session.id));
    const remove = document.createElement('button');
    remove.className = 'history-delete';
    remove.textContent = '×';
    remove.title = '删除对话';
    remove.addEventListener('click', async (event) => {
      event.stopPropagation();
      if (!confirm(`删除对话“${session.title}”？此操作只会删除客户端记录。`)) return;
      try {
        await window.naiwa.request('delete_session', { sessionId: session.id });
        await refreshSessions();
        if (state.sessionId === session.id) {
          state.sessionId = null;
          if (state.sessions.length) await selectSession(state.sessions[0].id);
          else await newChat();
        }
        toast('对话已删除');
      } catch (error) { toast(error.message, true); }
    });
    item.append(select, remove);
    list.append(item);
  }
}

async function refreshSessions() {
  const result = await window.naiwa.request('list_sessions');
  state.sessions = result.sessions || [];
  renderHistory();
}

function showWelcome() {
  const messages = $('#messages');
  messages.replaceChildren();
  const template = document.createElement('template');
  template.innerHTML = welcomeHTML;
  messages.append(template.content.firstElementChild);
  bindCapabilityCards();
}

const PROCESS_KIND_LABELS = {
  think: '思考',
  reasoning: '深度推理',
  tool_call: '调用工具',
  tool_result: '工具返回',
  notice: '系统'
};

// 与后端 agent/ui.py 的 strip_thinking 对齐：模型按提示词约定用【结果】标注
// 最终答案，但历史消息里可能整段存了"内心独白+答案"。只保留最后一个
// 【结果】之后的内容，让改造前保存的旧对话也能干净呈现。
function stripResultMark(text) {
  if (typeof text !== 'string') return text;
  const mark = '【结果】';
  if (!text.includes(mark)) return text;
  const tail = text.slice(text.lastIndexOf(mark) + mark.length).trim();
  return tail || text;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// 行内语法（输入必须已 escape）
function renderInlineMarkdown(text) {
  return text
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

// 轻量 markdown → HTML：标题、列表、表格、粗体、行内代码、代码块、链接。
// 目标是"正常对话"的排版，而不是把 ##、** 原样丢给用户。
function renderMarkdown(src) {
  if (!src) return '';
  const lines = escapeHtml(src).split('\n');
  const out = [];
  let para = [];
  let listType = null;
  let listItems = [];
  let tableRows = null;
  let inCode = false;
  let codeLines = [];

  const flushPara = () => {
    if (!para.length) return;
    out.push(`<p>${para.map(renderInlineMarkdown).join('<br>')}</p>`);
    para = [];
  };
  const flushList = () => {
    if (!listType) return;
    out.push(`<${listType}>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</${listType}>`);
    listType = null;
    listItems = [];
  };
  const flushTable = () => {
    if (!tableRows) return;
    const cells = (row) => row.replace(/^\||\|$/g, '').split('|').map((cell) => renderInlineMarkdown(cell.trim()));
    const head = `<tr>${cells(tableRows[0]).map((c) => `<th>${c}</th>`).join('')}</tr>`;
    const body = tableRows.slice(1).map((row) => `<tr>${cells(row).map((c) => `<td>${c}</td>`).join('')}</tr>`).join('');
    out.push(`<table>${head}${body}</table>`);
    tableRows = null;
  };
  const flushAll = () => { flushPara(); flushList(); flushTable(); };

  for (const line of lines) {
    if (/^```/.test(line)) {
      if (inCode) {
        out.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        flushAll();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeLines.push(line); continue; }
    const trimmed = line.trim();
    if (!trimmed) { flushAll(); continue; }
    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 1, 5); // ## → h3，避免标题过大
      out.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (tableRows && /^\|[\s:|-]+\|$/.test(trimmed)) continue; // 表格分隔行
    if (/^\|(.+)\|$/.test(trimmed)) {
      flushPara();
      flushList();
      if (!tableRows) tableRows = [];
      tableRows.push(trimmed);
      continue;
    }
    flushTable();
    const ul = trimmed.match(/^[-*+]\s+(.*)$/);
    if (ul) {
      flushPara();
      if (listType !== 'ul') { flushList(); listType = 'ul'; }
      listItems.push(ul[1]);
      continue;
    }
    const ol = trimmed.match(/^\d+[.、]\s*(.*)$/);
    if (ol) {
      flushPara();
      if (listType !== 'ol') { flushList(); listType = 'ol'; }
      listItems.push(ol[1]);
      continue;
    }
    flushList();
    para.push(trimmed);
  }
  if (inCode && codeLines.length) out.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
  flushAll();
  return out.join('');
}

function processItemElement(entry) {
  const item = document.createElement('div');
  const kind = entry.kind || 'think';
  const failed = kind === 'tool_result' && entry.status && entry.status !== 'success';
  item.className = `process-item ${kind}${failed ? ' error' : ''}`;
  item.dataset.kind = kind;
  const label = document.createElement('span');
  label.className = 'process-kind';
  label.textContent = kind === 'tool_call' || kind === 'tool_result'
    ? `${PROCESS_KIND_LABELS[kind]} · ${entry.name || ''}`
    : PROCESS_KIND_LABELS[kind] || kind;
  const text = document.createElement('div');
  text.className = 'process-text';
  // 旧会话的过程条目可能带【思考】前缀（新会话在 bridge 侧已剥）
  const raw = kind === 'tool_call' ? (entry.args || '') : (entry.text || '');
  text.textContent = String(raw).replace(/^\s*【思考】\s*/, '');
  item.append(label, text);
  return item;
}

// 思考与行动过程框：固定高度、可滚动；默认收起，答案在框外独立呈现（对齐后端 ProcessBox 设计）。
function processBoxElement(process, options = {}) {
  const box = document.createElement('div');
  box.className = `process-box${options.running ? ' running' : ' collapsed'}`;
  const head = document.createElement('button');
  head.type = 'button';
  head.className = 'process-head';
  const title = document.createElement('span');
  title.className = 'process-title';
  title.textContent = '思考与行动过程';
  const count = document.createElement('span');
  count.className = 'process-count';
  count.textContent = `${process.length} 步`;
  const arrow = document.createElement('span');
  arrow.className = 'process-arrow';
  arrow.textContent = options.running ? '▾' : '▸';
  head.append(title, count, arrow);
  const body = document.createElement('div');
  body.className = 'process-body';
  for (const entry of process) body.append(processItemElement(entry));
  head.addEventListener('click', () => {
    const collapsed = box.classList.toggle('collapsed');
    arrow.textContent = collapsed ? '▸' : '▾';
  });
  box.append(head, body);
  return { box, body, count, arrow };
}

function messageElement(role, content, time, usage, process, attachments) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  if (role === 'assistant') {
    const image = document.createElement('img');
    image.src = '../assets/naiwa-phone.png';
    image.alt = '奶蛙';
    avatar.append(image);
  } else avatar.textContent = '你';
  const body = document.createElement('div');
  body.className = 'message-body';
  const meta = document.createElement('div');
  meta.className = 'message-meta';
  const name = document.createElement('span');
  name.textContent = role === 'assistant' ? '奶蛙 Agent' : '你';
  const timestamp = document.createElement('time');
  timestamp.textContent = formatTime(time) || formatTime(new Date());
  meta.append(name, timestamp);
  const text = document.createElement('div');
  text.className = 'message-content';
  if (role === 'assistant') {
    // 答案按"正常对话"排版：markdown 渲染成标题/列表/表格/粗体；
    // stripResultMark 兜底清理改造前历史消息里混入的内心独白。
    text.innerHTML = renderMarkdown(stripResultMark(content || ''));
  } else {
    text.textContent = content || '';
  }
  body.append(meta);
  // 思考与行动过程框置于答案上方：答案干净独立，过程一键展开回看。
  if (role === 'assistant' && Array.isArray(process) && process.length) {
    body.append(processBoxElement(process).box);
  }
  body.append(text);
  // 用户消息的附件 chips（图片/文件名 + 大小，点击可打开本机文件）
  if (role === 'user' && Array.isArray(attachments) && attachments.length) {
    body.append(attachmentChipsElement(attachments));
  }
  if (usage) {
    const usageEl = document.createElement('div');
    usageEl.className = 'message-usage';
    usageEl.textContent = usage;
    body.append(usageEl);
  }
  row.append(avatar, body);
  return { row, text, body };
}

function appendMessage(role, content, time, usage, attachments) {
  let list = $('.message-list');
  if (!list) {
    $('#messages').replaceChildren();
    list = document.createElement('div');
    list.className = 'message-list';
    $('#messages').append(list);
  }
  const item = messageElement(role, content, time, usage, null, attachments);
  list.append(item.row);
  scrollMessages();
  return item;
}

function scrollMessages() {
  requestAnimationFrame(() => { $('#messages').scrollTop = $('#messages').scrollHeight; });
}

function renderSession(session) {
  state.sessionId = session.id;
  $('#chatTitle').textContent = session.title || '新对话';
  document.title = `${session.title || '新对话'} · 奶蛙 Agent`;
  if (!session.messages || !session.messages.length) showWelcome();
  else {
    $('#messages').replaceChildren();
    const list = document.createElement('div');
    list.className = 'message-list';
    $('#messages').append(list);
    for (const message of session.messages) {
      const item = messageElement(message.role, message.content, message.time, null, message.process, message.attachments);
      list.append(item.row);
    }
  }
  clearTrace();
  updateUsage(null);
  renderHistory();
  scrollMessages();
}

async function selectSession(sessionId) {
  if (state.sending && sessionId !== state.sessionId) {
    toast('当前任务还在运行，完成后再切换对话');
    return;
  }
  try {
    const session = await window.naiwa.request('get_session', { sessionId });
    renderSession(session);
    switchView('chat');
  } catch (error) { toast(error.message, true); }
}

async function newChat() {
  if (state.sending) {
    // 历史 bug：任务卡死时 newChat 被 guard 拦死，输入框永远 disabled。
    // 现在允许"停止当前任务并开新对话"，停止确认后继续。
    const confirmed = confirm('当前任务还在运行。\n停止当前任务并开启新对话？（未完成的部分回答会丢失）');
    if (!confirmed) return;
    await cancelRun();
    // 等后端确认取消（send_message 以 {cancelled:true} 收尾），最多 5 秒
    for (let i = 0; i < 50 && state.sending; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (state.sending) {
      toast('停止确认超时，请稍候再试', true);
      return;
    }
  }
  try {
    const session = await window.naiwa.request('new_session');
    await refreshSessions();
    renderSession(session);
    switchView('chat');
    // 双保险：无论之前发生过什么，新对话的输入框必须立即可用
    setRunning(false);
    $('#messageInput').focus();
  } catch (error) { toast(error.message, true); }
}

function renderModels() {
  const grid = $('#settingsModelGrid');
  const select = $('#headerModelSelect');
  grid.replaceChildren();
  select.replaceChildren();
  if (!state.models.length) {
    const empty = document.createElement('div');
    empty.className = 'models-empty';
    empty.innerHTML = '<span>⌁</span><strong>等待连接模型服务</strong><p>你的模型列表会出现在这里</p>';
    grid.append(empty);
    const option = document.createElement('option');
    option.value = '';
    option.textContent = '未配置模型';
    select.append(option);
    updateCurrentModel({});
    return;
  }
  for (const model of state.models) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.id;
    option.selected = model.current || model.id === state.currentModel.id;
    select.append(option);

    const card = document.createElement('button');
    card.className = `model-card${option.selected ? ' current' : ''}`;
    const top = document.createElement('div');
    top.className = 'model-card-top';
    const name = document.createElement('strong');
    name.textContent = model.id;
    name.title = model.label || model.id;
    const kind = document.createElement('span');
    kind.className = 'model-kind';
    kind.textContent = model.kind === 'reasoner' ? '深度推理' : '对话';
    top.append(name, kind);
    if (option.selected) {
      const current = document.createElement('span');
      current.className = 'model-current';
      current.textContent = '当前使用';
      top.append(current);
    }
    const description = document.createElement('p');
    description.textContent = model.description || (model.supports_tools ? '支持 Agent 工具调用' : '仅支持对话');
    card.append(top, description);
    if (model.supports_vision) {
      const vision = document.createElement('span');
      vision.className = 'model-kind vision';
      vision.textContent = '视觉';
      vision.title = '支持图片（多模态）输入';
      top.append(vision);
    }
    card.addEventListener('click', () => selectModel(model.id));
    grid.append(card);
  }
  $('#modelListDescription').textContent = `共 ${state.models.length} 个模型，点击卡片即可热切换，对话记忆会保留。`;
}

function updateCurrentModel(model) {
  state.currentModel = model || {};
  const name = model && model.id ? model.id : '未配置模型';
  $('#composerModel').textContent = name;
  $('#runModelName').textContent = model && model.id ? name : '尚未配置';
  if ($('#headerModelSelect').value !== (model.id || '')) $('#headerModelSelect').value = model.id || '';
  renderAttachments();   // 模型切换后刷新"不支持图片识别"提示
}

async function loadModels(provider = state.provider) {
  try {
    const result = await window.naiwa.request('models', { provider });
    state.models = result.models || [];
    updateCurrentModel(result.current || {});
    renderModels();
  } catch (error) {
    console.warn(error);
  }
}

async function selectModel(modelId) {
  if (!modelId || state.sending) return;
  try {
    const current = await window.naiwa.request('select_model', { modelId });
    state.models = state.models.map((model) => ({ ...model, current: model.id === modelId }));
    updateCurrentModel(current);
    renderModels();
    toast(`已切换到 ${modelId}，对话记忆保留`);
  } catch (error) { toast(error.message, true); }
}

async function configureAgent() {
  const button = $('#saveAndDiscover');
  const status = $('#configStatus');
  const apiKey = $('#apiKeyInput').value.trim();
  if (!apiKey && !state.hasKey) {
    toast('请先输入 API Key', true);
    $('#apiKeyInput').focus();
    return;
  }
  button.disabled = true;
  button.textContent = '正在连接…';
  status.className = 'config-status';
  status.textContent = `正在验证 ${PROVIDERS[state.provider].label} 并拉取可用模型，请稍候…`;
  try {
    const settings = await window.naiwa.saveSettings({ provider: state.provider, apiKey });
    state.hasKey = settings.hasKey;
    const result = await window.naiwa.configureAgent(state.provider);
    state.configured = true;
    state.models = result.models || [];
    updateCurrentModel(result.current || {});
    renderModels();
    $('#toolCount').textContent = result.tools || '—';
    $('#keyStateDot').classList.add('ready');
    $('#apiKeyInput').value = '';
    $('#apiKeyInput').placeholder = '密钥已保存；输入新 Key 可替换';
    status.textContent = `连接成功：接口返回 ${result.discovered} 个模型，当前展示 ${result.models.length} 个。`;
    $('#backendStatus').textContent = '本地 Agent 已就绪';
    $('#backendDot').className = 'status-pulse ready';
    toast(`已获取 ${result.discovered} 个可用模型`);
  } catch (error) {
    state.configured = false;
    status.className = 'config-status error';
    status.textContent = `连接失败：${error.message}`;
    toast('模型连接失败，请检查 Key 与网络', true);
  } finally {
    button.disabled = false;
    button.textContent = '保存并获取模型';
  }
}

function setRunning(running) {
  $('#runState').textContent = running ? '执行中' : '待命中';
  $('#runState').classList.toggle('running', running);
  $('#sendButton').disabled = running;
  $('#sendButton').classList.toggle('hidden', running);
  $('#stopButton').classList.toggle('hidden', !running);
  $('#messageInput').disabled = running;
}

// 停止当前任务：通知后端取消；UI 状态由 send_message 的 {cancelled:true} 响应收尾
async function cancelRun() {
  if (state.runTimer) {
    clearTimeout(state.runTimer);
    state.runTimer = null;
  }
  if (!state.sending || !state.sessionId) return;
  try {
    await window.naiwa.request('cancel', { sessionId: state.sessionId });
  } catch (error) {
    toast(`停止失败：${error.message}`, true);
  }
}

function addTrace(type, title, detail = '', isError = false) {
  const trace = $('#toolTrace');
  const empty = trace.querySelector('.trace-empty');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = `trace-item${isError ? ' error' : ''}`;
  const strong = document.createElement('strong');
  strong.textContent = title;
  const small = document.createElement('small');
  small.textContent = `${type} · ${formatTime(new Date())}`;
  item.append(strong, small);
  if (detail) {
    const p = document.createElement('p');
    p.textContent = detail;
    item.append(p);
  }
  trace.prepend(item);
  while (trace.children.length > 30) trace.lastElementChild.remove();
}

function clearTrace() {
  $('#toolTrace').innerHTML = '<div class="trace-empty">发送任务后，工具调用与执行过程会显示在这里。</div>';
}

function updateUsage(sessionUsage, turnUsage) {
  if (!sessionUsage) {
    $('#sessionTokens').textContent = '0 tokens';
    $('#sessionCost').textContent = '¥0.0000';
    return;
  }
  $('#sessionTokens').textContent = `${(sessionUsage.total_tokens || 0).toLocaleString()} tokens`;
  $('#sessionCost').textContent = `约 ¥${(sessionUsage.cost || 0).toFixed(4)}`;
  if (turnUsage && state.streamElement) {
    const old = state.streamElement.body.querySelector('.message-usage');
    if (old) old.remove();
    const el = document.createElement('div');
    el.className = 'message-usage';
    const total = turnUsage.total_tokens || ((turnUsage.input_tokens || 0) + (turnUsage.output_tokens || 0));
    el.textContent = `本次${turnUsage.estimated ? '估算' : '实耗'} ${total.toLocaleString()} tokens · 模型调用 ${turnUsage.calls || 0} 次`;
    state.streamElement.body.append(el);
  }
}

async function sendMessage() {
  const input = $('#messageInput');
  const message = input.value.trim();
  if ((!message && !state.attachments.length) || state.sending) return;
  if (!state.configured) {
    switchView('settings');
    toast('先配置 API Key 并获取模型，再开始对话');
    return;
  }
  if (!state.sessionId) await newChat();
  if (!state.sessionId) return;   // 新建会话失败：保持原状（含附件），不进入发送态
  const sessionId = state.sessionId;
  state.sending = true;
  setRunning(true);
  input.value = '';
  input.style.height = 'auto';
  clearTrace();
  $('#estimateBar').classList.add('hidden');
  const attachments = state.attachments.map((item) => ({
    uploadId: item.uploadId, name: item.name, mime: item.mime, size: item.size
  }));
  state.attachments = [];
  renderAttachments();
  appendMessage('user', message || `（发送 ${attachments.length} 个附件）`,
    new Date().toISOString(), null, attachments);
  state.streamElement = appendMessage('assistant', '', new Date().toISOString());
  state.streamElement.text.classList.add('pending');
  // 思考与行动过程框：流式期间展开并实时跟进，完成后自动收起，答案独立呈现。
  const process = processBoxElement([], { running: true });
  state.streamElement.body.insertBefore(process.box, state.streamElement.text);
  state.streamElement.process = process;
  addTrace('任务', '奶蛙已接收任务', (message || '（仅附件）').slice(0, 180));
  // 看门狗：后端硬超时之外的双保险，确保任何情况下输入框都不会被永久锁死
  state.runTimer = setTimeout(() => cancelRun(), (state.runTimeoutSeconds + 120) * 1000);
  try {
    const result = await window.naiwa.request('send_message', { sessionId, message, attachments });
    if (result.cancelled) {
      // 手动停止 / 超时取消：UI 恢复，答案不落库（用户消息已由后端保存）
      const timeout = result.reason === 'timeout';
      state.streamElement.text.classList.remove('pending');
      state.streamElement.text.textContent = timeout
        ? `任务超时（${state.runTimeoutSeconds} 秒）已自动停止，输入框已恢复，可重试。`
        : '任务已手动停止。';
      state.streamElement.row.classList.add('error');
      finishStreamProcess();
      addTrace(timeout ? '超时' : '停止',
        timeout ? '任务超时已停止' : '任务已手动停止',
        timeout ? `超过 ${state.runTimeoutSeconds} 秒未完成，已自动取消` : '用户手动停止了当前任务', true);
      await refreshSessions().catch(() => {});
      const stopped = state.sessions.find((session) => session.id === sessionId);
      if (stopped) $('#chatTitle').textContent = stopped.title;
    } else {
      state.streamElement.text.innerHTML = renderMarkdown(result.answer || '');
      state.streamElement.text.classList.remove('pending');
      finishStreamProcess();
      updateUsage(result.sessionUsage, result.usage);
      await refreshSessions();
      const current = state.sessions.find((session) => session.id === sessionId);
      if (current) $('#chatTitle').textContent = current.title;
      addTrace('完成', '任务执行完成', result.visionUnsupported
        ? '当前模型不支持图片识别，已返回降级提示（未调用模型）'
        : '最终回答已写入当前对话');
    }
  } catch (error) {
    state.streamElement.row.classList.add('error');
    state.streamElement.text.textContent = `任务未完成：${error.message}`;
    state.streamElement.text.classList.remove('pending');
    finishStreamProcess();
    addTrace('错误', '任务执行失败', error.message, true);
    await refreshSessions().catch(() => {});
  } finally {
    if (state.runTimer) {
      clearTimeout(state.runTimer);
      state.runTimer = null;
    }
    state.sending = false;
    state.streamElement = null;
    stopToolTimer();
    setRunning(false);
    $('#estimateBar').classList.add('hidden');
    input.focus();
    scrollMessages();
  }
}

// 工具调用计时：pip install / 跑测试这类命令常要几十秒，过程框若一直静止
// 用户会以为卡死。每秒把已用时间挂在工具名后面，收到 tool_result 立刻停表。
// 超过 30 秒补一句提示——长命令是正常现象，也可以点「停止」中断。
function startToolTimer(item, name) {
  stopToolTimer();
  const label = item && item.querySelector('.process-kind');
  if (!label) return;
  const base = label.dataset.base || label.textContent;
  label.dataset.base = base;
  const started = Date.now();
  const tick = () => {
    const seconds = Math.round((Date.now() - started) / 1000);
    label.textContent = seconds >= 30
      ? `${base} · ${seconds}s（较慢，可点「停止」中断）`
      : `${base} · ${seconds}s`;
  };
  tick();
  state.toolTimer = setInterval(tick, 1000);
}

function stopToolTimer() {
  if (state.toolTimer) clearInterval(state.toolTimer);
  state.toolTimer = null;
  // 计时文字挂在「调用工具 · 名字」这个 label 上：摘掉后缀恢复原样
  document.querySelectorAll('.process-item.tool_call .process-kind').forEach((el) => {
    if (el.dataset.base) el.textContent = el.dataset.base;
  });
}

// 流式过程条目进入当前消息的过程框：连续的同类文本合并为一条，避免碎片节点。
function appendProcessEntry(entry) {
  const process = state.streamElement && state.streamElement.process;
  if (!process) return;
  const kind = entry.kind || 'think';
  const last = process.body.lastElementChild;
  if ((kind === 'think' || kind === 'reasoning') && last && last.dataset.kind === kind) {
    last.querySelector('.process-text').textContent += entry.text || '';
  } else {
    process.body.append(processItemElement(entry));
  }
  process.count.textContent = `${process.body.children.length} 步`;
  process.body.scrollTop = process.body.scrollHeight;
  scrollMessages();
  // 右侧运行面板保持实时状态：推理与工具事件同步推送。
  if (kind === 'reasoning') {
    addTrace('推理', '模型正在思考', (entry.text || '').slice(-400));
  } else if (kind === 'tool_call') {
    let args = entry.args || '';
    try { args = JSON.stringify(JSON.parse(entry.args), null, 2); } catch { args = String(entry.args || ''); }
    addTrace('调用工具', entry.name || '未知工具', String(args).slice(0, 1200));
    startToolTimer(process.body.lastElementChild, entry.name);
  } else if (kind === 'tool_result') {
    stopToolTimer();
    addTrace(entry.status === 'success' ? '工具完成' : '工具异常', entry.name || '工具', entry.text || '', entry.status !== 'success');
  }
}

function finishStreamProcess() {
  const process = state.streamElement && state.streamElement.process;
  if (!process) return;
  stopToolTimer();
  process.box.classList.remove('running');
  process.box.classList.add('collapsed');
  process.arrow.textContent = '▸';
}

// 清空当前消息的过程框（后端判定输出退化后重跑时用）
function clearStreamProcess() {
  const process = state.streamElement && state.streamElement.process;
  if (!process) return;
  stopToolTimer();
  process.body.replaceChildren();
  process.count.textContent = '0 步';
}

function handleAgentEvent(event) {
  if (event.type === 'ready') {
    $('#backendStatus').textContent = '本地 Agent 已连接';
    $('#backendDot').className = 'status-pulse ready';
    return;
  }
  if (event.sessionId && event.sessionId !== state.sessionId) return;
  if (event.type === 'process') {
    appendProcessEntry(event);
  } else if (event.type === 'process_reset') {
    // 后端判定本轮输出退化（复读成「好。执行。好。执行。」）并中止／重跑：
    // 已推过来的垃圾思考没有保留价值，清空过程框，只留重跑后的干净内容。
    clearStreamProcess();
  } else if (event.type === 'estimate') {
    const estimate = event.estimate;
    $('#estimateBar').textContent = `预计约 ${(estimate.total_tokens || 0).toLocaleString()} tokens · ${estimate.expected_calls || 1} 次模型调用`;
    $('#estimateBar').classList.remove('hidden');
  } else if (event.type === 'session') {
    refreshSessions().catch(() => {});
  }
}

function renderJournal() {
  const list = $('#journalList');
  list.replaceChildren();
  $('#journalCount').textContent = state.journalEntries.length;
  if (!state.journalEntries.length) {
    const empty = document.createElement('div');
    empty.className = 'journal-empty';
    empty.textContent = '还没有工作日记。记录今天的进展，给明天留一个清晰的起点。';
    list.append(empty);
    return;
  }
  const entries = [...state.journalEntries].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  for (const entry of entries) {
    const card = document.createElement('div');
    card.className = `journal-card${entry.id === state.journalId ? ' active' : ''}`;
    const time = document.createElement('time');
    time.textContent = entry.date;
    const title = document.createElement('h3');
    title.textContent = entry.title;
    const content = document.createElement('p');
    content.textContent = entry.content || '（空白日记）';
    const footer = document.createElement('div');
    footer.className = 'journal-card-footer';
    const count = document.createElement('span');
    count.textContent = `${entry.content.length} 字`;
    const remove = document.createElement('button');
    remove.className = 'journal-delete';
    remove.textContent = '删除';
    remove.addEventListener('click', async (event) => {
      event.stopPropagation();
      if (!confirm(`删除日记“${entry.title}”？`)) return;
      await window.naiwa.deleteJournal(entry.id);
      if (state.journalId === entry.id) resetJournalEditor();
      await refreshJournal();
      toast('日记已删除');
    });
    footer.append(count, remove);
    card.append(time, title, content, footer);
    card.addEventListener('click', () => editJournal(entry));
    list.append(card);
  }
}

async function refreshJournal() {
  state.journalEntries = await window.naiwa.listJournal();
  renderJournal();
}

function editJournal(entry) {
  state.journalId = entry.id;
  $('#journalDate').value = entry.date;
  $('#journalTitle').value = entry.title;
  $('#journalContent').value = entry.content;
  $('#journalEditorLabel').textContent = '编辑过往日记';
  renderJournal();
}

function resetJournalEditor() {
  state.journalId = null;
  $('#journalDate').value = localDate();
  $('#journalTitle').value = '';
  $('#journalContent').value = '';
  $('#journalEditorLabel').textContent = '今天的心得';
  $('#journalSaveState').textContent = '本地保存';
  renderJournal();
  $('#journalTitle').focus();
}

async function saveJournal() {
  const title = $('#journalTitle').value.trim();
  const content = $('#journalContent').value.trim();
  if (!title && !content) {
    toast('先写一点内容再保存', true);
    return;
  }
  try {
    const previous = state.journalEntries.find((entry) => entry.id === state.journalId);
    const saved = await window.naiwa.saveJournal({
      id: state.journalId,
      createdAt: previous?.createdAt,
      date: $('#journalDate').value || localDate(),
      title: title || '工作记录',
      content
    });
    state.journalId = saved.id;
    $('#journalSaveState').textContent = '已保存';
    await refreshJournal();
    toast('日记已保存到本机');
  } catch (error) { toast(error.message, true); }
}

async function insertChatSummary() {
  if (!state.sessionId) return;
  try {
    const session = await window.naiwa.request('get_session', { sessionId: state.sessionId });
    const messages = (session.messages || []).slice(-6);
    if (!messages.length) {
      toast('当前对话还没有可带入的内容');
      return;
    }
    const summary = [`对话主题：${session.title}`, ...messages.map((message) => `${message.role === 'user' ? '我' : '奶蛙'}：${message.content}`)].join('\n\n');
    const field = $('#journalContent');
    field.value = `${field.value}${field.value ? '\n\n' : ''}${summary}`;
    if (!$('#journalTitle').value) $('#journalTitle').value = `工作记录 · ${session.title}`.slice(0, 80);
    toast('已带入当前对话摘要');
  } catch (error) { toast(error.message, true); }
}

function bindCapabilityCards() {
  $$('.capability-card').forEach((card) => card.addEventListener('click', () => {
    $('#messageInput').value = card.dataset.prompt;
    $('#messageInput').focus();
    $('#messageInput').dispatchEvent(new Event('input'));
  }));
}

function bindEvents() {
  $$('.nav-item').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.view)));
  $('#newChatTop').addEventListener('click', newChat);
  $('#newChatHeader').addEventListener('click', newChat);
  $('#sendButton').addEventListener('click', sendMessage);
  $('#stopButton').addEventListener('click', () => cancelRun());
  $('#attachButton').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', (event) => {
    attachFiles(event.target.files);
    event.target.value = '';   // 清空，允许再次选择同一文件
  });
  // 粘贴上传：剪贴板里的图片/文件直接作为附件
  $('#messageInput').addEventListener('paste', (event) => {
    const files = event.clipboardData && event.clipboardData.files;
    if (files && files.length) {
      event.preventDefault();
      attachFiles(files);
    }
  });
  // 拖拽上传：整个对话区可放；preventDefault 防止 Electron 把文件当导航打开
  const chatView = $('#chatView');
  let dragDepth = 0;
  const hasFiles = (event) => Boolean(event.dataTransfer) &&
    [...event.dataTransfer.types].includes('Files');
  chatView.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    $('#composer').classList.add('drag-active');
  });
  chatView.addEventListener('dragover', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  chatView.addEventListener('dragleave', (event) => {
    event.preventDefault();
    dragDepth = Math.max(dragDepth - 1, 0);
    if (!dragDepth) $('#composer').classList.remove('drag-active');
  });
  chatView.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    $('#composer').classList.remove('drag-active');
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) {
      switchView('chat');
      attachFiles(files);
    }
  });
  document.addEventListener('dragover', (event) => event.preventDefault());
  document.addEventListener('drop', (event) => event.preventDefault());
  $('#messageInput').addEventListener('input', (event) => {
    event.target.style.height = 'auto';
    event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`;
  });
  $('#messageInput').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.key.toLowerCase() === 'n') {
      event.preventDefault();
      newChat();
    }
  });
  $('#themeToggle').addEventListener('click', () => applyTheme(state.theme === 'day' ? 'night' : 'day'));
  $$('.theme-option').forEach((item) => item.addEventListener('click', () => applyTheme(item.dataset.themeChoice)));
  $$('.provider-tab').forEach((item) => item.addEventListener('click', () => updateProvider(item.dataset.provider)));
  $('#saveAndDiscover').addEventListener('click', configureAgent);
  $('#apiKeyInput').addEventListener('keydown', (event) => { if (event.key === 'Enter') configureAgent(); });
  $('#toggleKeyVisibility').addEventListener('click', () => {
    const input = $('#apiKeyInput');
    input.type = input.type === 'password' ? 'text' : 'password';
    $('#toggleKeyVisibility').textContent = input.type === 'password' ? '显示' : '隐藏';
  });
  $('#openKeyPage').addEventListener('click', () => window.naiwa.openExternal(PROVIDERS[state.provider].keyUrl));
  $('#headerModelSelect').addEventListener('change', (event) => selectModel(event.target.value));
  $('#clearTrace').addEventListener('click', clearTrace);
  $('#newJournal').addEventListener('click', resetJournalEditor);
  $('#saveJournal').addEventListener('click', saveJournal);
  $('#insertChatSummary').addEventListener('click', insertChatSummary);
  $$('.window-controls button').forEach((button) => button.addEventListener('click', () => window.naiwa.windowControl(button.dataset.window)));
  let laughTimer;
  let finishTimer;
  const closeIntro = () => {
    clearTimeout(laughTimer);
    clearTimeout(finishTimer);
    const intro = $('#intro');
    const video = $('#laughVideo');
    if (video) video.pause();
    if (!intro) return;
    intro.classList.add('leaving');
    setTimeout(() => intro.remove(), 700);
  };
  const startLaugh = () => {
    const intro = $('#intro');
    const video = $('#laughVideo');
    if (!intro) return;
    intro.classList.add('laughing');
    if (video) {
      try {
        video.currentTime = 0;
        video.volume = 0.8;
        const playback = video.play();
        if (playback) {
          playback.catch(() => {
            video.muted = true;
            video.play().catch(() => {});
          });
        }
      } catch {}
    }
    finishTimer = setTimeout(closeIntro, 4000);
  };
  $('#skipIntro').addEventListener('click', closeIntro);
  laughTimer = setTimeout(startLaugh, 4300);
  window.naiwa.onAgentEvent(handleAgentEvent);
  bindCapabilityCards();
}

async function init() {
  bindEvents();
  resetJournalEditor();
  try {
    const settings = await window.naiwa.getSettings();
    state.provider = settings.provider || 'deepseek';
    state.hasKey = settings.hasKey;
    applyTheme(settings.theme || 'day');
    $$('.provider-tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.provider === state.provider));
    $('#apiKeyLabel').textContent = `${PROVIDERS[state.provider].label} API Key`;
    if (state.hasKey) {
      $('#keyStateDot').classList.add('ready');
      $('#apiKeyInput').placeholder = '密钥已保存；输入新 Key 可替换';
    }
    const health = await window.naiwa.request('health');
    $('#toolCount').textContent = health.tools || '—';
    state.runTimeoutSeconds = health.run_timeout_seconds || 900;
    $('#backendStatus').textContent = '本地 Agent 已连接';
    $('#backendDot').className = 'status-pulse ready';
    await refreshSessions();
    if (state.sessions.length) await selectSession(state.sessions[0].id);
    else await newChat();
    await refreshJournal();
    if (state.hasKey) await configureAgent();
  } catch (error) {
    $('#backendStatus').textContent = '本地 Agent 连接异常';
    $('#backendDot').className = 'status-pulse error';
    toast(error.message, true);
  }
}

init();
