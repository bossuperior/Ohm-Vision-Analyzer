import tkinter as tk
from src.inference.model_engine import YOLOPoseEngine
from src.inference.pipeline import IntelligentBreadboardPipeline
from vision.camera_loader import CameraLoader


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Panel")
        self.root.geometry("200x100")

        # 1. Initialize Backend Systems
        print("Initializing AI and Camera...")
        self.camera = CameraLoader(camera_id=1)  # Or 0 depending on your system
        self.camera.start()

        self.engine = YOLOPoseEngine("models/Yolo_v8n_pose_weights.onnx")
        self.pipeline = IntelligentBreadboardPipeline(self.camera, self.engine)

        # 2. Setup UI
        self.close_button = tk.Button(
            root, text="Close Camera",
            command=self.on_closing, bg="red", fg="white", font=("Arial", 12)
        )
        self.close_button.pack(expand=True, fill='both', padx=20, pady=20)

        # 3. Start the loop
        self.update_video()

    def update_video(self):
        """This replaces the while True loop. It runs one frame, then schedules the next."""
        self.pipeline.process_single_frame()

        # Schedule this function to run again in 10 milliseconds
        self.root.after(30, self.update_video)

    def on_closing(self):
        """Safe shutdown."""
        print("Shutting down...")
        self.camera.stop()
        self.engine.release_resources()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    # Bind the window's close (X) button to our safe shutdown method
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()