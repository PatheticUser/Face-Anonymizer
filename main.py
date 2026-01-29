import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print("Program started")

if not cap.isOpened():
    print("Camera not detected")
    exit()

colors = {
    "Green": ([35, 50, 50], [85, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Red1": ([0, 120, 70], [10, 255, 255]),
    "Red2": ([170, 120, 70], [180, 255, 255]),
    "White": ([0, 0, 200], [180, 40, 255])
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR → HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green color range (tunable)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Apply mask to original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours of green objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore tiny noise areas
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Object", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Green Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
