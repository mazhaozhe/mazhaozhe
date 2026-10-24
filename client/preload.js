const { contextBridge, ipcRenderer } = require('electron');

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
  windowControl: (action) => ipcRenderer.send('window:control', action),
  platform: process.platform
});
