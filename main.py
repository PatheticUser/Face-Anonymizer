import cv2
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -------------------------------
# YOLO11 Model (nano)
# -------------------------------
model = YOLO("yolo11n.pt")  # nano model for CPU

# -------------------------------
# Webcam capture thread
# -------------------------------
class WebcamStream:
    def __init__(self, src=0, width=640, height=360):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.cap.release()

# -------------------------------
# Anonymization functions
# -------------------------------
def pixelate(face_roi, blocks=15):
    h, w = face_roi.shape[:2]
    temp = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def blur(face_roi, ksize=15):
    k = ksize if ksize % 2 == 1 else ksize+1
    return cv2.GaussianBlur(face_roi, (k, k), 0)

def blackbox(face_roi):
    return 0 * face_roi

# -------------------------------
# Main GUI App
# -------------------------------
class FaceAnonymizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Anonymizer")
        self.root.geometry("700x500")
        self.root.configure(bg="#222222")

        # Video display
        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        # Controls
        self.controls_frame = tk.Frame(root, bg="#333333")
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.start_btn = ttk.Button(self.controls_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.controls_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.toggle_btn = ttk.Button(self.controls_frame, text="Toggle Anonymization", command=self.toggle_anonymization)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        # Mode selector
        self.mode_var = tk.StringVar(value="pixelate")
        self.mode_menu = ttk.OptionMenu(self.controls_frame, self.mode_var, "pixelate", "pixelate", "blur", "blackbox")
        self.mode_menu.pack(side=tk.LEFT, padx=5)

        # Pixelation / blur slider
        self.slider = ttk.Scale(self.controls_frame, from_=5, to=50, orient=tk.HORIZONTAL, command=self.update_slider)
        self.slider.set(15)
        self.slider.pack(side=tk.LEFT, padx=5)
        self.slider_label = tk.Label(self.controls_frame, text="Value: 15", bg="#333333", fg="white")
        self.slider_label.pack(side=tk.LEFT, padx=5)

        # State
        self.stream = None
        self.running = False
        self.anonymize = True
        self.frame_count = 0
        self.last_results = None
        self.slider_value = 15

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_camera(self):
        if self.stream is None:
            self.stream = WebcamStream()
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.stream:
            self.stream.release()
            self.stream = None
            self.video_label.config(image='')

    def toggle_anonymization(self):
        self.anonymize = not self.anonymize

    def update_slider(self, val):
        self.slider_value = int(float(val))
        self.slider_label.config(text=f"Value: {self.slider_value}")

    def update_frame(self):
        if self.running and self.stream:
            ret, frame = self.stream.read()
            if ret:
                frame_resized = cv2.resize(frame, (640, 360))

                # YOLO every 3 frames
                if self.frame_count % 3 == 0:
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    self.last_results = model(frame_rgb)[0]

                annotated_frame = frame_resized.copy()

                if self.last_results is not None and self.anonymize:
                    boxes = self.last_results.boxes.xyxy.cpu().numpy()
                    classes = self.last_results.boxes.cls.cpu().numpy()
                    for i, cls in enumerate(classes):
                        if int(cls) == 0:
                            x1, y1, x2, y2 = boxes[i].astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(annotated_frame.shape[1]-1, x2), min(annotated_frame.shape[0]-1, y2)
                            face_roi = annotated_frame[y1:y2, x1:x2]
                            if face_roi.size == 0:
                                continue
                            mode = self.mode_var.get()
                            if mode == "pixelate":
                                anonymized = pixelate(face_roi, blocks=self.slider_value)
                            elif mode == "blur":
                                anonymized = blur(face_roi, ksize=self.slider_value)
                            else:
                                anonymized = blackbox(face_roi)
                            annotated_frame[y1:y2, x1:x2] = anonymized

                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                self.frame_count += 1

            self.root.after(10, self.update_frame)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnonymizerApp(root)
    root.mainloop()
