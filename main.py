import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectors.HSVDetector import HSVDetector
from utils.roi_selector import select_roi

video_path = "videos/Video1_husky.mp4"
roi = select_roi(video_path)
cap = cv2.VideoCapture(video_path)

output_video_path = "output/trajectory.mp4"

detector_type = "HSV"

detectors = {
    "HSV": HSVDetector,
}

if detector_type == "HSV":
    lower_hsv = (0, 70, 50)
    upper_hsv = (10, 255, 255)
    detector = detectors[detector_type](lower_hsv, upper_hsv)
else:
    raise ValueError(f"Detector type '{detector_type}' is not supported.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_width, frame_height = 640, 480
fps = cap.get(cv2.CAP_PROP_FPS) or 30 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

poses = []
trajectory = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    box, rect, mask = detector.detect(frame, roi)

    if box is not None:
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        if rect:
            center = tuple(map(int, rect[0]))
            angle = rect[2]
            poses.append((frame_count, center[0], center[1], angle))
            trajectory.append(center)

    for point in trajectory:
        cv2.circle(frame, point, 2, (255, 0, 0), -1)

    cv2.imshow('Frame', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

frames, x_positions, y_positions, angles = zip(*poses)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(frames, x_positions, label='Posição X')
plt.ylabel('X (pixels)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(frames, y_positions, label='Posição Y')
plt.ylabel('Y (pixels)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(frames, angles, label='Ângulo (graus)')
plt.xlabel('Número de amostras')
plt.ylabel('Ângulo')
plt.legend()

plt.tight_layout()
plt.savefig("output/amostras.png")
