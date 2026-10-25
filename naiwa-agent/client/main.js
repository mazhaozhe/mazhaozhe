const { app, BrowserWindow, ipcMain, safeStorage, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..');
const PYTHON = path.join(ROOT, '.venv', 'Scripts', 'python.exe');
const BRIDGE = path.join(__dirname, 'bridge.py');
const ALLOWED_METHODS = new Set([
  'health', 'models', 'select_model', 'new_session', 'list_sessions',
  'get_session', 'delete_session', 'send_message', 'tools',
  'upload_file', 'discard_upload', 'cancel'
]);
let mainWindow;
let bridge;
let requestSeq = 0;
let runtimeKey = '';
const pending = new Map();
const SMOKE_TEST = process.env.NAIWA_SMOKE_TEST === '1';
const consoleErrors = [];
// 开启动画的大笑视频需要自动播放；Electron 默认可能拦截带声音的自动播放。
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
if (SMOKE_TEST) {
  app.setPath('userData', path.join(app.getPath('temp'), 'naiwa-agent-smoke-test'));
}

const files = {
  settings: () => path.join(app.getPath('userData'), 'settings.json'),
  journal: () => path.join(app.getPath('userData'), 'journal.json')
};

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch { return fallback; }
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temp = `${file}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value, null, 2), 'utf8');
  fs.renameSync(temp, file);
}

function publicSettings() {
  const saved = readJson(files.settings(), {});
  return {
    provider: saved.provider || 'deepseek',
    theme: saved.theme || 'day',
    hasKey: Boolean(saved.encryptedKey || runtimeKey),
    keyProtection: saved.encryptedKey ? 'encrypted' : (runtimeKey ? 'memory' : 'none')
  };
}

function getApiKey() {
  if (runtimeKey) return runtimeKey;
  const saved = readJson(files.settings(), {});
  if (!saved.encryptedKey) return '';
  if (!safeStorage.isEncryptionAvailable()) return '';
  try { return safeStorage.decryptString(Buffer.from(saved.encryptedKey, 'base64')); }
  catch { return ''; }
}

function sendToBridge(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!bridge || bridge.killed || !bridge.stdin.writable) {
      reject(new Error('本地 Agent 服务尚未启动，请重启客户端'));
      return;
    }
    const id = ++requestSeq;
    pending.set(id, { resolve, reject });
    bridge.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
  });
}

function startBridge() {
  if (!fs.existsSync(PYTHON)) {
    throw new Error('未找到项目 .venv，请先按原项目说明安装 Python 依赖');
  }
  bridge = spawn(PYTHON, [BRIDGE], {
    cwd: ROOT,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
      ...(SMOKE_TEST ? { CLIENT_SESSION_FILE: path.join(app.getPath('temp'), 'naiwa-agent-smoke-sessions.json') } : {})
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true
  });
  const lines = readline.createInterface({ input: bridge.stdout });
  lines.on('line', (line) => {
    let message;
    try { message = JSON.parse(line); } catch { return; }
    if (message.event) {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('agent:event', message.data ? { type: message.event, ...message.data } : { type: message.event });
      }
      return;
    }
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id);
    if (message.ok) item.resolve(message.result);
    else item.reject(new Error(message.error || 'Agent 请求失败'));
  });
  bridge.stderr.on('data', (chunk) => console.error(`[agent] ${chunk.toString()}`));
  bridge.on('exit', () => {
    for (const item of pending.values()) item.reject(new Error('本地 Agent 服务已退出'));
    pending.clear();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 980,
    minHeight: 680,
    show: false,
    frame: false,
    backgroundColor: '#eef2ea',
    title: '奶蛙 Agent',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.webContents.on('console-message', (_event, level, message) => {
    if (level >= 3) consoleErrors.push(message);
  });
  mainWindow.once('ready-to-show', () => {
    if (SMOKE_TEST) runSmokeTest();
    else mainWindow.show();
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
}

async function runSmokeTest() {
  try {
    await new Promise((resolve) => setTimeout(resolve, 1800));
    const result = await mainWindow.webContents.executeJavaScript(`(async () => {
      const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      await wait(3500);
      const intro = document.querySelector('#intro');
      const laughVideo = document.querySelector('#laughVideo');
      const introLaughing = Boolean(intro && intro.classList.contains('laughing'));
      const introFrogUsesNewPose = document.querySelector('.frog-character img').src.includes('naiwa-intro.png');
      const laughVideoLoaded = Boolean(laughVideo && laughVideo.readyState >= 2);
      const laughVideoDuration = laughVideo ? laughVideo.duration : 0;
      const laughVideoPlaying = Boolean(laughVideo && !laughVideo.paused && laughVideo.currentTime > 0.1);
      const laughCaptionRemoved = !document.querySelector('.laugh-caption');
      const laughVideoStyle = laughVideo ? getComputedStyle(laughVideo) : null;
      const laughVideoFullBleed = Boolean(laughVideoStyle && laughVideoStyle.objectFit === 'cover' &&
        laughVideo.offsetWidth >= innerWidth - 1 && laughVideo.offsetHeight >= innerHeight - 1);
      document.querySelector('#skipIntro').click();
      for (let i = 0; i < 120; i += 1) {
        const status = document.querySelector('#backendStatus').textContent;
        if (status.includes('已连接')) break;
        await wait(100);
      }
      await wait(300);
      // 附件上传：文件选择 → 暂存 → 附件 chips → 移除
      const attachButtonExists = Boolean(document.querySelector('#attachButton'));
      const fileInputExists = Boolean(document.querySelector('#fileInput'));
      const attachmentTrayExists = Boolean(document.querySelector('#attachmentTray'));
      const stopButtonExists = Boolean(document.querySelector('#stopButton'));
      const uploadSample = new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])],
        'naiwa.png', { type: 'image/png' });
      const uploadTransfer = new DataTransfer();
      uploadTransfer.items.add(uploadSample);
      const fileInputEl = document.querySelector('#fileInput');
      fileInputEl.files = uploadTransfer.files;
      fileInputEl.dispatchEvent(new Event('change', { bubbles: true }));
      let uploadChipAppeared = false;
      for (let i = 0; i < 80; i += 1) {
        await wait(100);
        if (document.querySelector('#attachmentTray .attachment-chip')) {
          uploadChipAppeared = true;
          break;
        }
      }
      const uploadRemove = document.querySelector('#attachmentTray .attachment-chip .chip-remove');
      if (uploadRemove) uploadRemove.click();
      await wait(300);
      const uploadChipRemoved = !document.querySelector('#attachmentTray .attachment-chip');
      // 历史 bug 回归：开启新对话后输入框必须立即可用（发送中被锁死过）
      document.querySelector('#newChatHeader').click();
      let newChatInputEnabled = false;
      for (let i = 0; i < 60; i += 1) {
        await wait(100);
        if (!document.querySelector('#messageInput').disabled) {
          newChatInputEnabled = true;
          break;
        }
      }
      document.querySelector('[data-view="settings"]').click();
      document.querySelector('[data-theme-choice="night"]').click();
      const nightApplied = document.documentElement.dataset.theme === 'night';
      await wait(500);
      const nightWallpaper = getComputedStyle(document.querySelector('.wallpaper'));
      const nightWallpaperUsesImage = nightWallpaper.backgroundImage.includes('night-umbrella.jpg');
      const nightWallpaperIsFullBleed = nightWallpaper.backgroundSize === 'cover';
      const nightPanelStyle = getComputedStyle(document.querySelector('.main-content')).backgroundImage;
      const nightPanelIsTranslucent = nightPanelStyle.includes('rgba(3, 8, 11');
      const previewWidth = document.querySelector('.theme-preview').getBoundingClientRect().width;
      document.querySelector('[data-theme-choice="day"]').click();
      const dayApplied = document.documentElement.dataset.theme === 'day';
      const dayWallpaper = getComputedStyle(document.querySelector('.wallpaper'));
      const dayWallpaperUsesImage = dayWallpaper.backgroundImage.includes('day-kungfu.jpg');
      const dayPanelStyle = getComputedStyle(document.querySelector('.main-content')).backgroundImage;
      const dayPanelIsTranslucent = dayPanelStyle.includes('rgba(253, 252, 244');
      const welcomeAvatarRemoved = !document.querySelector('.welcome-avatar');
      const brandUsesPhoneFrog = document.querySelector('.titlebar-brand img').src.includes('naiwa-phone.png');
      const providerCount = document.querySelectorAll('.provider-tab').length;
      document.querySelector('[data-view="journal"]').click();
      const journalVisible = document.querySelector('#journalView').classList.contains('active');
      document.querySelector('[data-view="chat"]').click();
      const chatVisible = document.querySelector('#chatView').classList.contains('active');
      // 思考与行动过程框：结构、可滚动、默认收起、可展开、答案与过程分离
      const processSample = messageElement('assistant', '这是最终答案', new Date().toISOString(), null, [
        { kind: 'think', text: '先梳理任务要点，再决定是否需要调用工具。' },
        { kind: 'tool_call', name: 'list_files', args: '{"path":"."}' },
        { kind: 'tool_result', name: 'list_files', status: 'success', text: 'ok' }
      ]);
      document.querySelector('#messages').append(processSample.row);
      const processBox = processSample.row.querySelector('.process-box');
      const processBody = processSample.row.querySelector('.process-body');
      const processBodyStyle = processBody ? getComputedStyle(processBody) : null;
      const processBoxScrollable = Boolean(processBodyStyle &&
        processBodyStyle.overflowY === 'auto' && parseFloat(processBodyStyle.maxHeight) > 0);
      const processBoxCollapsedByDefault = Boolean(processBox && processBox.classList.contains('collapsed'));
      const answerSeparatedFromProcess = processSample.text.textContent === '这是最终答案' &&
        !processSample.text.textContent.includes('先梳理任务要点');
      processSample.row.querySelector('.process-head').click();
      const processBoxExpandsOnClick = Boolean(processBox && !processBox.classList.contains('collapsed'));
      const processItemCount = processSample.row.querySelectorAll('.process-item').length;
      processSample.row.remove();
      // markdown 渲染 + 答案纯净度 + 正文字体
      const mdSample = messageElement('assistant',
        '【思考】这段内心独白不该出现在答案里。【结果】## 交付说明\\n\\n**snake_plus.html** 已保存：\\n\\n- 平滑移动\\n- 粒子特效\\n\\n| 版本 | 特性 |\\n|---|---|\\n| 旧版 | 经典 |\\n| 新版 | 增强 |',
        new Date().toISOString());
      document.querySelector('#messages').append(mdSample.row);
      const mdHtml = mdSample.text.innerHTML;
      const markdownRendered = mdHtml.includes('<strong>') && mdHtml.includes('<h3>') &&
        mdHtml.includes('<table>') && mdHtml.includes('<ul>') &&
        !mdHtml.includes('##') && !mdHtml.includes('**') && !mdHtml.includes('内心独白');
      const mdStyle = getComputedStyle(mdSample.text);
      const answerFontReadable = parseFloat(mdStyle.fontSize) >= 14 && Number(mdStyle.fontWeight) >= 600;
      mdSample.row.remove();
      const brokenImages = [...document.images].filter((img) => img.complete && img.naturalWidth === 0).map((img) => img.src);
      const composer = document.querySelector('.composer').getBoundingClientRect();
      const chat = document.querySelector('#chatView').getBoundingClientRect();
      return {
        title: document.title,
        viewport: [innerWidth, innerHeight],
        noHorizontalOverflow: document.documentElement.scrollWidth <= innerWidth,
        composerInsideView: composer.bottom <= chat.bottom + 1,
        introLaughing,
        introFrogUsesNewPose,
        laughVideoLoaded,
        laughVideoDuration,
        laughVideoPlaying,
        laughCaptionRemoved,
        laughVideoFullBleed,
        nightApplied,
        nightWallpaperUsesImage,
        nightWallpaperImage: nightWallpaper.backgroundImage,
        nightWallpaperIsFullBleed,
        nightPanelIsTranslucent,
        previewWidth,
        dayApplied,
        dayWallpaperUsesImage,
        dayWallpaperImage: dayWallpaper.backgroundImage,
        dayPanelIsTranslucent,
        welcomeAvatarRemoved,
        brandUsesPhoneFrog,
        providerCount,
        journalVisible,
        chatVisible,
        processBoxScrollable,
        processBoxCollapsedByDefault,
        answerSeparatedFromProcess,
        processBoxExpandsOnClick,
        processItemCount,
        markdownRendered,
        answerFontReadable,
        attachButtonExists,
        fileInputExists,
        attachmentTrayExists,
        stopButtonExists,
        uploadChipAppeared,
        uploadChipRemoved,
        newChatInputEnabled,
        historyCount: Number(document.querySelector('#historyCount').textContent),
        backend: document.querySelector('#backendStatus').textContent,
        brokenImages
      };
    })()`);
    result.consoleErrors = consoleErrors;
    result.pass = result.noHorizontalOverflow && result.composerInsideView &&
      result.introLaughing && result.introFrogUsesNewPose && result.laughVideoLoaded &&
      result.laughVideoDuration >= 3.8 && result.laughVideoPlaying &&
      result.laughCaptionRemoved && result.laughVideoFullBleed &&
      result.nightApplied && result.nightWallpaperUsesImage && result.nightWallpaperIsFullBleed &&
      result.nightPanelIsTranslucent && result.previewWidth >= 130 &&
      result.dayApplied && result.dayWallpaperUsesImage && result.dayPanelIsTranslucent &&
      result.welcomeAvatarRemoved && result.brandUsesPhoneFrog && result.providerCount === 3 &&
      result.journalVisible && result.chatVisible && result.historyCount >= 1 &&
      result.processBoxScrollable && result.processBoxCollapsedByDefault &&
      result.answerSeparatedFromProcess && result.processBoxExpandsOnClick &&
      result.processItemCount === 3 &&
      result.markdownRendered && result.answerFontReadable &&
      result.attachButtonExists && result.fileInputExists &&
      result.attachmentTrayExists && result.stopButtonExists &&
      result.uploadChipAppeared && result.uploadChipRemoved &&
      result.newChatInputEnabled &&
      result.brokenImages.length === 0 && consoleErrors.length === 0;
    console.log(`SMOKE_RESULT ${JSON.stringify(result)}`);
    app.exit(result.pass ? 0 : 1);
  } catch (error) {
    console.error(`SMOKE_RESULT ${JSON.stringify({ pass: false, error: error.message, consoleErrors })}`);
    app.exit(1);
  }
}

app.whenReady().then(() => {
  startBridge();
  createWindow();

  ipcMain.handle('settings:get', () => publicSettings());
  ipcMain.handle('settings:save', (_event, input) => {
    const previous = readJson(files.settings(), {});
    const next = { ...previous, provider: input.provider || previous.provider || 'deepseek' };
    if (typeof input.theme === 'string') next.theme = input.theme;
    if (input.apiKey && input.apiKey.trim()) {
      runtimeKey = input.apiKey.trim();
      if (safeStorage.isEncryptionAvailable()) {
        next.encryptedKey = safeStorage.encryptString(runtimeKey).toString('base64');
      } else {
        delete next.encryptedKey;
      }
    }
    writeJson(files.settings(), next);
    return publicSettings();
  });
  ipcMain.handle('agent:configure', async (_event, provider) => {
    const apiKey = getApiKey();
    if (!apiKey) throw new Error('请先输入并保存 API Key；若系统加密不可用，本次运行期间会仅保存在内存中');
    return sendToBridge('configure', { provider, apiKey });
  });
  ipcMain.handle('bridge:request', (_event, { method, params }) => {
    if (!ALLOWED_METHODS.has(method)) throw new Error('不支持的客户端请求');
    return sendToBridge(method, params || {});
  });
  ipcMain.handle('journal:list', () => readJson(files.journal(), { entries: [] }).entries || []);
  ipcMain.handle('journal:save', (_event, entry) => {
    const data = readJson(files.journal(), { entries: [] });
    const now = new Date().toISOString();
    const item = {
      id: entry.id || `${Date.now()}-${Math.random().toString(16).slice(2, 7)}`,
      title: String(entry.title || '工作记录').slice(0, 80),
      content: String(entry.content || '').slice(0, 20000),
      date: entry.date || now.slice(0, 10),
      updatedAt: now,
      createdAt: entry.createdAt || now
    };
    const index = data.entries.findIndex((old) => old.id === item.id);
    if (index >= 0) data.entries[index] = item;
    else data.entries.unshift(item);
    writeJson(files.journal(), data);
    return item;
  });
  ipcMain.handle('journal:delete', (_event, id) => {
    const data = readJson(files.journal(), { entries: [] });
    data.entries = data.entries.filter((entry) => entry.id !== id);
    writeJson(files.journal(), data);
    return { deleted: id };
  });
  ipcMain.handle('app:open-external', (_event, url) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url);
  });
  // 打开本地上传目录里的附件（仅限 data/uploads，防路径穿越）
  ipcMain.handle('app:open-path', async (_event, target) => {
    if (typeof target !== 'string' || !target) return '';
    const uploadsRoot = path.resolve(ROOT, 'data', 'uploads');
    const resolved = path.resolve(target);
    if (resolved !== uploadsRoot && !resolved.startsWith(uploadsRoot + path.sep)) return '';
    return shell.openPath(resolved);
  });
  ipcMain.on('window:control', (_event, action) => {
    if (!mainWindow) return;
    if (action === 'minimize') mainWindow.minimize();
    if (action === 'maximize') mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
    if (action === 'close') mainWindow.close();
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
}).catch((error) => {
  console.error(error);
  app.quit();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (bridge && !bridge.killed) bridge.kill();
});
