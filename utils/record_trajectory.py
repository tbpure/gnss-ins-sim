import tkinter as tk
from tkinter import messagebox
import csv


class DebugTrajectoryDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("鼠标轨迹记录（带强制刷新）")

        # 1. 界面布局
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white', highlightthickness=1, relief="sunken")
        self.canvas.pack(padx=20, pady=20)

        self.info_label = tk.Label(root, text="操作指南：按住鼠标左键涂鸦，点击下方按钮保存", fg="gray")
        self.info_label.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        tk.Button(self.btn_frame, text="保存轨迹为 CSV", command=self.save_to_csv, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="清空画布", command=self.clear_canvas, width=10).pack(side=tk.LEFT, padx=5)

        # 2. 数据状态
        self.all_paths = []
        self.current_path = []
        self.last_x, self.last_y = None, None

        # 3. 绑定事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.last_x, self.last_y = event.x, event.y
        self.current_path = [(event.x, event.y)]
        # 在起点画一个小点，确保点击也有反馈
        self.canvas.create_oval(event.x - 1, event.y - 1, event.x + 1, event.y + 1, fill="red", outline="red")

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            # 绘制线条
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="red", width=3, capstyle=tk.ROUND, smooth=True, tags="line"
            )

            # 记录坐标
            self.current_path.append((event.x, event.y))
            self.last_x, self.last_y = event.x, event.y

            # 强制画布实时更新显示
            self.canvas.update()

    def stop_drawing(self, event):
        if self.current_path:
            self.all_paths.append(list(self.current_path))
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.all_paths = []  # 清空已记录的路径

    def save_to_csv(self):
        if not self.all_paths:
            messagebox.showwarning("警告", "当前没有任何轨迹数据")
            return

        try:
            with open("trajectory_data.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Stroke_ID", "X", "Y"])
                for i, stroke in enumerate(self.all_paths):
                    for pt in stroke:
                        writer.writerow([i + 1, pt[0], pt[1]])
            messagebox.showinfo("成功", "轨迹已成功保存至 trajectory_data.csv")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    # 窗口置顶，防止被其他窗口遮挡导致不刷新
    root.attributes('-topmost', True)
    app = DebugTrajectoryDrawer(root)
    root.mainloop()