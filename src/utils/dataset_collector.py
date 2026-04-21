import cv2
import tkinter as tk
import os
from PIL import Image, ImageTk

class DataCollectorApp:
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("480x640")
        self.video_source = video_source

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        self.save_dir = os.path.join(project_root, "data", "raw", "dataset_capture")
        os.makedirs(self.save_dir, exist_ok=True)

        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.vid.isOpened():
            print("⚠️ Camera not opened!")

        self.create_widgets()
        #Live Preview
        self.update_frame()

        # Close Event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind('<space>', lambda e: self.capture_image())

    def create_widgets(self):
        self.header = tk.Label(self.window, text="Ohm-Vision Data Collector", font=("Arial", 20, "bold"))
        self.header.pack(pady=10)

        # Live Preview
        self.canvas = tk.Canvas(self.window, bg="black", highlightthickness=0)
        self.canvas.pack(pady=5, expand=True, fill="both")

        self.status_label = tk.Label(self.window, text="Ready...", fg="blue", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.btn_frame = tk.Frame(self.window)
        self.btn_frame.pack(pady=20)

        self.btn_capture = tk.Button(
            self.window, text="📸 Capture Image (Spacebar)",
            command=self.capture_image,
            bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
            width=25, height=2
        )
        self.btn_capture.pack(pady=10)

        self.btn_close = tk.Button(
            self.window, text="❌ Close",
            command=self.on_closing,
            bg="#f44336", fg="white", width=15
        )
        self.btn_close.pack(pady=5)

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                frame_resized = cv2.resize(frame, (canvas_width, canvas_height))
                self.frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update_frame)

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
            count = len([f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]) + 1
            filename = os.path.join(self.save_dir, f"img_{count:03d}.jpg")
            cv2.imwrite(filename, frame)

            self.status_label.config(text=f" Saved: img_{count:03d}.jpg", fg="green")
            print(f" Captured: {filename}")

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root, "Ohm Vision Data Collector", video_source=1)
    root.mainloop()