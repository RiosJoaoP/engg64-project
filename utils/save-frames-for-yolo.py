import cv2
import os

VIDEO_NAME = "video2_husky"

def save_yolo_label(x1, y1, x2, y2, img_width, img_height, label_file):
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    label_file.write(f"0 {x_center} {y_center} {width} {height}\n")

def process_video(video_path, output_dir):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Desenhe a caixa do objeto. Pressione 'Enter' para finalizar.")
            bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            frame_filename = os.path.join(images_dir, f"{VIDEO_NAME}_frame_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)

            label_filename = os.path.join(labels_dir, f"{VIDEO_NAME}_frame_{frame_id}.txt")
            with open(label_filename, 'w') as label_file:
                save_yolo_label(x1, y1, x2, y2, frame.shape[1], frame.shape[0], label_file)

            print(f"Anotação salva para o frame {frame_id}")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

video_path = f"videos/{VIDEO_NAME}.mp4"
output_dir = "output/yolo_frames"

process_video(video_path, output_dir)
