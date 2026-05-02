import cv2
import tkinter as tk
import os
from PIL import Image, ImageTk
from src.vision.breadboard_warper import BreadboardWarper


class DataCollectorApp:
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x680")
        self.video_source = video_source

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        self.save_dir = os.path.join(project_root, "data", "raw", "dataset_capture")
        os.makedirs(self.save_dir, exist_ok=True)

        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.vid.isOpened():
            print("Camera not opened!")

        self.warper = BreadboardWarper(output_width=810, output_height=540)
        self.current_warped = None  # warped frame พร้อม save

        self.create_widgets()
        self.update_frame()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind('<space>', lambda e: self.capture_image())

    def create_widgets(self):
        self.header = tk.Label(self.window,
                               text="Ohm-Vision Data Collector (Warped)",
                               font=("Arial", 16, "bold"))
        self.header.pack(pady=8)

        self.canvas = tk.Canvas(self.window, bg="black",
                                width=810, height=540, highlightthickness=0)
        self.canvas.pack(pady=4)

        self.status_label = tk.Label(self.window, text="Searching for ArUco tags...",
                                     fg="red", font=("Arial", 11, "bold"))
        self.status_label.pack(pady=4)

        self.btn_capture = tk.Button(
            self.window, text="Capture Warped Frame  [Spacebar]",
            command=self.capture_image,
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
            width=30, height=2
        )
        self.btn_capture.pack(pady=6)

        self.count_label = tk.Label(self.window, text="Saved: 0 images",
                                    font=("Arial", 10))
        self.count_label.pack()

        self.btn_close = tk.Button(self.window, text="Close",
                                   command=self.on_closing,
                                   bg="#f44336", fg="white", width=15)
        self.btn_close.pack(pady=6)

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            success, warped, _ = self.warper.process(frame)

            if success:
                self.current_warped = warped.copy()
                display = warped.copy()
                cv2.putText(display, "BOARD OK — Ready to capture",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2)
                self.status_label.config(text="Board detected — Press SPACE to capture",
                                         fg="green")
                self.btn_capture.config(state=tk.NORMAL, bg="#4CAF50")
            else:
                self.current_warped = None
                display = cv2.resize(frame, (810, 540))
                cv2.putText(display, "SEARCHING FOR ARUCO TAGS...",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                self.status_label.config(text="Searching for ArUco tags...", fg="red")
                self.btn_capture.config(state=tk.DISABLED, bg="#9E9E9E")

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update_frame)

    def capture_image(self):
        if self.current_warped is None:
            self.status_label.config(text="Cannot capture — board not detected!", fg="red")
            return

        existing = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        count = len(existing) + 1
        filename = os.path.join(self.save_dir, f"img_{count:03d}.jpg")
        cv2.imwrite(filename, self.current_warped)

        self.count_label.config(text=f"Saved: {count} images")
        self.status_label.config(text=f"Saved: img_{count:03d}.jpg", fg="blue")
        print(f"Captured: {filename}")

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root, "Ohm Vision Data Collector", video_source=1)
    root.mainloop()
