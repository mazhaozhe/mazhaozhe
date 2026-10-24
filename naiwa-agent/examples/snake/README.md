# examples/snake —— Agent 自己写的贪吃蛇

这三个文件不是手写的示例代码，而是 **GeneralTaskAgent 在开发过程中自己写出来、自己跑测试验证** 的真实产物，用来展示它的"代码开发 + 自我验证"能力：

| 文件 | 说明 |
|------|------|
| `snake_game.py` | tkinter 版贪吃蛇，零第三方依赖，`python snake_game.py` 直接玩（方向键/WASD 移动，空格暂停，R 重开，Esc 退出） |
| `snake.html` | 单文件 HTML 经典版，浏览器打开即玩 |
| `snake_plus.html` | 增强版：平滑移动、粒子特效、加速机制 |
| `_test_snake_plus.js` | Agent 为自己写的轻量浏览器环境 mock 测试（`node _test_snake_plus.js`），在无浏览器环境下验证 `snake_plus.html` 的核心逻辑 |

运行方式：

```powershell
python snake_game.py          # tkinter 版
node _test_snake_plus.js      # 跑它给自己写的测试
```
