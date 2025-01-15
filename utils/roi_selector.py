import cv2

def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not load video.")
        cap.release()
        exit()

    frame = cv2.resize(frame, (640, 480))
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    cap.release()

    if roi == (0, 0, 0, 0):
        print("No ROI selected.")
        exit()

    return int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3])
