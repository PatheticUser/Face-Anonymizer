import cv2
import threading
from ultralytics import YOLO

# -------------------------------
# YOLO11 Model (nano)
# -------------------------------
# Note: use a YOLO11 model that includes 'person' class; we will filter faces only.
# If you have a face-only YOLO model, use that instead.
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
# Anonymization function (pixelate)
# -------------------------------
def pixelate(face_roi, blocks=15):
    # Shrink face
    h, w = face_roi.shape[:2]
    temp = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    # Scale back to original size
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

# -------------------------------
# Initialize webcam
# -------------------------------
stream = WebcamStream()
frame_skip = 3   # YOLO every 3 frames
frame_count = 0
last_results = None

print("YOLO Face Anonymizer started")

while True:
    ret, frame = stream.read()
    if not ret:
        continue

    # Resize frame for speed
    frame_resized = cv2.resize(frame, (640, 360))

    # Run YOLO every N frames
    if frame_count % frame_skip == 0:
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)[0]
        last_results = results

    annotated_frame = frame_resized.copy()

    # Process YOLO detections
    if last_results is not None:
        boxes = last_results.boxes.xyxy.cpu().numpy()  # xyxy format
        classes = last_results.boxes.cls.cpu().numpy()  # class IDs
        for i, cls in enumerate(classes):
            # Filter only 'person' class (YOLO COCO class 0)
            if int(cls) == 0:
                x1, y1, x2, y2 = boxes[i].astype(int)
                # Clip coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(annotated_frame.shape[1]-1, x2), min(annotated_frame.shape[0]-1, y2)
                face_roi = annotated_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                # Pixelate face
                anonymized_face = pixelate(face_roi, blocks=15)
                annotated_frame[y1:y2, x1:x2] = anonymized_face

    cv2.imshow("YOLO Face Anonymizer", annotated_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
