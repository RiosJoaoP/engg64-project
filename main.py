import cv2
import matplotlib.pyplot as plt
from detectors.HSVDetector import HSVDetector
from utils.roi_selector import select_roi
import numpy as np
import collections

video_name = "video2_husky"
video_path = f"videos/{video_name}.mp4"
roi = select_roi(video_path)
cap = cv2.VideoCapture(video_path)

output_video_path = f"output/trajectory_{video_name}.mp4"

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

frame_width, frame_height = 640, 480
fps = cap.get(cv2.CAP_PROP_FPS) or 30 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

poses = []
trajectory = []

angle_history = collections.deque(maxlen=5)
last_angle = None

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    box, data, mask = detector.detect(frame, roi)

    if box is not None and data is not None:
        cx, cy, angle = data

        if last_angle is not None:
            diff = abs(angle - last_angle)
            if diff > 20:
                angle = last_angle

        angle_history.append(angle)
        smoothed_angle = sum(angle_history) / len(angle_history)
        last_angle = smoothed_angle

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        angle_text = f"Angle: {smoothed_angle:.1f}"
        cv2.putText(
            frame, angle_text, (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

        poses.append((frame_count, cx, cy, smoothed_angle))
        trajectory.append((cx, cy))

    for point in trajectory:
        cv2.circle(frame, point, 2, (255, 0, 0), -1)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb_resized = cv2.resize(mask_rgb, (frame.shape[1], frame.shape[0]))
    combined_frame = np.hstack((frame, mask_rgb_resized))

    cv2.imshow('Frame', combined_frame)

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
plt.savefig(f"output/amostras_{video_name}.png")
