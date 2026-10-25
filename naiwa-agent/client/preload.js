const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('naiwa', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (settings) => ipcRenderer.invoke('settings:save', settings),
  configureAgent: (provider) => ipcRenderer.invoke('agent:configure', provider),
  request: (method, params = {}) => ipcRenderer.invoke('bridge:request', { method, params }),
  onAgentEvent: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on('agent:event', listener);
    return () => ipcRenderer.removeListener('agent:event', listener);
  },
  listJournal: () => ipcRenderer.invoke('journal:list'),
  saveJournal: (entry) => ipcRenderer.invoke('journal:save', entry),
  deleteJournal: (id) => ipcRenderer.invoke('journal:delete', id),
  openExternal: (url) => ipcRenderer.invoke('app:open-external', url),
  // 打开本地上传目录里的附件（main.js 侧限制在 data/uploads 内）
  openPath: (target) => ipcRenderer.invoke('app:open-path', target),
  // 取 File 的本机绝对路径（Electron 扩展，失败返回空串，由调用方走 base64 兜底）
  getPathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file) || '';
    } catch {
      return '';
    }
  },
  windowControl: (action) => ipcRenderer.send('window:control', action),
  platform: process.platform
});
