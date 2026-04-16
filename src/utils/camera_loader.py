import cv2
import threading
import time


class CameraLoader:
    """Handles webcam hardware on a dedicated background thread for zero-lag I/O."""

    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None

        # Threading variables
        self.ret = False
        self.frame = None
        self.is_running = False
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam ID: {self.camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Read the first frame to initialize
        self.ret, self.frame = self.cap.read()

        # Start the background thread
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"📸 Threaded Camera {self.camera_id} started at {actual_w}x{actual_h}")

    def _update_loop(self):
        """Continuously grabs frames in the background."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                # Prevent CPU spin if camera drops a frame
                time.sleep(0.01)

    def get_frame(self):
        """Instantly returns the most recent frame from memory."""
        if not self.ret or self.frame is None:
            return False, None
        return self.ret, self.frame.copy()

    def stop(self):
        """Safely shuts down the thread and hardware."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join()  # Wait for thread to finish
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Camera released.")