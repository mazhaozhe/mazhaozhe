"""贪吃蛇小游戏（tkinter 实现，零第三方依赖）

运行：python snake_game.py
操作：
    方向键 / WASD  移动
    空格           暂停 / 继续
    R              重新开始
    Esc            退出
"""

import random
import tkinter as tk
from tkinter import messagebox

# ---------------- 可调参数 ----------------
CELL = 24               # 每格像素
COLS, ROWS = 24, 24     # 棋盘格数
BASE_SPEED = 140        # 初始每步毫秒数（越小越快）
MIN_SPEED = 60          # 最快速度上限
SPEED_UP_EVERY = 5      # 每吃几个食物提速一次

BG = "#1e1e2e"
GRID = "#2a2a3c"
SNAKE_HEAD = "#a6e3a1"
SNAKE_BODY = "#74c7a1"
FOOD = "#f38ba8"
TEXT = "#cdd6f4"

UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)


class SnakeGame:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("贪吃蛇")
        self.root.resizable(False, False)

        self.width = COLS * CELL
        self.height = ROWS * CELL
        self.top_bar = 40

        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height + self.top_bar,
            bg=BG,
            highlightthickness=0,
        )
        self.canvas.pack()

        self.high_score = 0
        self.after_id = None
        self.paused = False
        self.running = False

        self.root.bind("<Key>", self.on_key)
        self.reset()

    # ---------------- 游戏状态 ----------------
    def reset(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        cx, cy = COLS // 2, ROWS // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = RIGHT
        self.pending_direction = RIGHT
        self.score = 0
        self.speed = BASE_SPEED
        self.paused = False
        self.running = True
        self.food = self.spawn_food()
        self.draw()
        self.schedule()

    def spawn_food(self):
        empty = [
            (x, y)
            for x in range(COLS)
            for y in range(ROWS)
            if (x, y) not in self.snake
        ]
        return random.choice(empty) if empty else None

    def schedule(self):
        self.after_id = self.root.after(self.speed, self.tick)

    # ---------------- 主循环 ----------------
    def tick(self):
        self.after_id = None
        if not self.running or self.paused:
            return

        # 应用方向（禁止 180 度掉头）
        if (self.pending_direction[0] + self.direction[0],
                self.pending_direction[1] + self.direction[1]) != (0, 0):
            self.direction = self.pending_direction

        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # 撞墙
        if not (0 <= new_head[0] < COLS and 0 <= new_head[1] < ROWS):
            return self.game_over()

        # 撞自己（尾巴即将移开的那格不算）
        body = self.snake[:-1] if new_head != self.food else self.snake
        if new_head in body:
            return self.game_over()

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            if self.score % SPEED_UP_EVERY == 0:
                self.speed = max(MIN_SPEED, self.speed - 10)
            self.food = self.spawn_food()
            if self.food is None:      # 通关：铺满棋盘
                self.draw()
                return self.game_over(win=True)
        else:
            self.snake.pop()

        self.draw()
        self.schedule()

    def game_over(self, win=False):
        self.running = False
        if self.score > self.high_score:
            self.high_score = self.score
        self.draw()
        title = "通关！" if win else "游戏结束"
        again = messagebox.askyesno(
            title, f"{title}\n本局得分：{self.score}\n最高分：{self.high_score}\n\n再来一局？"
        )
        if again:
            self.reset()
        else:
            self.root.destroy()

    # ---------------- 绘制 ----------------
    def draw(self):
        c = self.canvas
        c.delete("all")

        # 顶部信息栏
        c.create_text(
            10, self.top_bar // 2, anchor="w", fill=TEXT,
            font=("Consolas", 13, "bold"),
            text=f"得分: {self.score}    最高分: {self.high_score}",
        )
        if self.paused:
            c.create_text(
                self.width - 10, self.top_bar // 2, anchor="e",
                fill=FOOD, font=("Consolas", 13, "bold"), text="已暂停 (空格继续)",
            )

        off = self.top_bar

        # 网格
        for x in range(COLS + 1):
            c.create_line(x * CELL, off, x * CELL, off + self.height, fill=GRID)
        for y in range(ROWS + 1):
            c.create_line(0, off + y * CELL, self.width, off + y * CELL, fill=GRID)

        # 食物
        if self.food:
            fx, fy = self.food
            pad = 4
            c.create_oval(
                fx * CELL + pad, off + fy * CELL + pad,
                (fx + 1) * CELL - pad, off + (fy + 1) * CELL - pad,
                fill=FOOD, outline="",
            )

        # 蛇
        for i, (x, y) in enumerate(self.snake):
            pad = 2
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            c.create_rectangle(
                x * CELL + pad, off + y * CELL + pad,
                (x + 1) * CELL - pad, off + (y + 1) * CELL - pad,
                fill=color, outline="",
            )

    # ---------------- 输入 ----------------
    def on_key(self, event):
        key = event.keysym.lower()
        mapping = {
            "up": UP, "w": UP,
            "down": DOWN, "s": DOWN,
            "left": LEFT, "a": LEFT,
            "right": RIGHT, "d": RIGHT,
        }
        if key in mapping:
            self.pending_direction = mapping[key]
        elif key == "space":
            if self.running:
                self.paused = not self.paused
                if not self.paused and self.after_id is None:
                    self.schedule()
                self.draw()
        elif key == "r":
            self.reset()
        elif key == "escape":
            self.root.destroy()


def main():
    root = tk.Tk()
    SnakeGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
