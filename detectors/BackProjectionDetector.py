import cv2
import numpy as np

from detectors.Detector import Detector

class BackprojectionDetector(Detector):
    def __init__(self, roi_hist):
        """
        Inicializa o detector com o histograma da ROI.

        Args:
            roi_hist: Histograma da região de interesse em HSV.
        """
        super().__init__()
        self.roi_hist = roi_hist

    def detect(self, frame, roi=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Backprojection
        dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dst = cv2.filter2D(dst, -1, kernel)

        # Threshold and mask
        _, mask = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            return box, rect, mask

        return None, None, mask
