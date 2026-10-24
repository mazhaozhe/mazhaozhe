// 轻量浏览器环境 mock，用于验证 snake_plus.html 的核心逻辑
const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('snake_plus.html', 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

let NOW = 0;
const performance = { now: () => NOW };

let rafCb = null, rafId = 1;
const requestAnimationFrame = (cb) => { rafCb = cb; return rafId++; };
const cancelAnimationFrame = () => { rafCb = null; };

const gradient = { addColorStop() {} };
const ctxTarget = {};
const ctx = new Proxy(ctxTarget, {
  get(t, p) {
    if (p === 'createRadialGradient' || p === 'createLinearGradient') return () => gradient;
    if (p in t) return t[p];
    return () => {};
  },
  set(t, p, v) { t[p] = v; return true; }
});

function makeEl(id) {
  return {
    id, textContent: '', innerHTML: '', style: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); },
      remove(c) { this._s.delete(c); },
      toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); }
    },
    addEventListener(ev, fn) { (this._h ||= {})[ev] = fn; },
    getContext() { return ctx; },
    width: 0, height: 0
  };
}
const els = {};
const document = { getElementById(id) { return (els[id] ||= makeEl(id)); } };

const winHandlers = {};
const window = {
  innerWidth: 1200, innerHeight: 900, devicePixelRatio: 1,
  addEventListener(ev, fn) { winHandlers[ev] = fn; },
  AudioContext: undefined, webkitAudioContext: undefined
};

const store = {};
const localStorage = {
  getItem: k => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); }
};

// 关键：Math.random 固定为 0 → 食物必然生成在第一个空格 (0,0)
const MyMath = Object.create(Math);
MyMath.random = () => 0;

const sandbox = {
  document, window, localStorage, performance,
  requestAnimationFrame, cancelAnimationFrame,
  Math: MyMath, Object, Array, JSON, console, Set, Map,
  isNaN, parseInt, parseFloat, String, Number, Boolean
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

// ================= 驱动工具 =================
function key(k) { winHandlers.keydown({ key: k, preventDefault() {} }); }
function frames(n, dtMs = 20) {
  for (let i = 0; i < n; i++) {
    if (!rafCb) return i;
    const cb = rafCb; rafCb = null;
    NOW += dtMs;
    cb(NOW);
  }
  return n;
}
const $score = els['score'];
const $ov = els['overlay'];
const $ovTitle = els['ovTitle'];
const $btnWall = els['btnWall'];
const isOver = () => !$ov.classList.contains('hidden');
const score = () => parseInt($score.textContent) || 0;

// 精确走 n 步：每步 130ms，每帧 20ms → 7 帧(140ms)走 1 步
function advanceSteps(n) {
  for (let i = 0; i < n; i++) {
    if (isOver()) return i;
    frames(7, 20);
  }
  return n;
}

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log('  ✓', name); }
  else { fail++; console.log('  ✗', name, extra); }
}

console.log('--- 初始状态 ---');
check('初始分数为 0', score() === 0, `实际=${$score.textContent}`);
check('初始显示遮罩层', isOver());

console.log('--- 按方向键开始 ---');
key('ArrowRight');
check('开始后遮罩隐藏', !isOver());

console.log('--- 撞墙结束（食物在(0,0)，向右走不会吃到）---');
let g = 0;
while (!isOver() && g < 2000) { frames(1, 20); g++; }
check('撞墙后游戏结束', isOver(), `guard=${g}`);
check('标题为"游戏结束"', $ovTitle.textContent === '游戏结束', `实际=${$ovTitle.textContent}`);
check('撞墙时分数仍为 0', score() === 0, `实际=${score()}`);

console.log('--- 吃食物：蛇从(11,11)向上11步到(11,0)，再向左11步到(0,0) ---');
key('r');
check('重开后分数归零', score() === 0);
key('ArrowUp');
advanceSteps(11);             // 向上走 11 步到 (11,0)
check('向上走完未结束', !isOver());
key('ArrowLeft');
advanceSteps(11);             // 向左走 11 步，到达 (0,0) 吃食物
check('吃到食物后分数 > 0', score() > 0, `实际=${score()}`);

console.log('--- 最高分持久化 ---');
// 让游戏结束一次，触发 best 写入
let g2 = 0;
while (!isOver() && g2 < 2000) { frames(1, 20); g2++; }
check('游戏已结束', isOver());
check('localStorage 写入最高分', 'snake2_best' in store, `keys=${Object.keys(store)}`);
check('最高分等于本局得分', store['snake2_best'] === String(score()), `存储=${store['snake2_best']} 本局=${score()}`);

console.log('--- 暂停 / 继续 ---');
key('r');
key('ArrowRight');
frames(3, 20);
key(' ');
check('空格后进入暂停', $ovTitle.textContent === '已暂停', `实际=${$ovTitle.textContent}`);
key(' ');
check('再按空格继续', !isOver());

console.log('--- 穿墙开关 ---');
$btnWall._h.click();
check('点击后穿墙开启', $btnWall.textContent.includes('开'), `实际=${$btnWall.textContent}`);
$btnWall._h.click();
check('再点关闭', $btnWall.textContent.includes('关'), `实际=${$btnWall.textContent}`);

console.log('--- 穿墙生效：向右走超过 22 步不结束 ---');
key('r');
$btnWall._h.click();          // 开启穿墙
key('ArrowRight');
frames(30 * 8, 20);           // 走 30 步，超过 22 列
check('穿墙模式下走 30 步未撞墙结束', !isOver(), `分数=${score()}`);

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail ? 1 : 0);
