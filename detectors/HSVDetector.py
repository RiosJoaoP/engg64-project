import cv2
import numpy as np
from detectors.Detector import Detector

class HSVDetector(Detector):
    def __init__(self, lower_hsv, upper_hsv):
        """
        Inicializa o detector HSV.
        
        :param lower_hsv: Tupla (H, S, V) para o limite inferior da cor.
        :param upper_hsv: Tupla (H, S, V) para o limite superior da cor.
        """
        self.lower_hsv1 = np.array(lower_hsv)
        self.upper_hsv1 = np.array(upper_hsv)
        self.lower_hsv2 = np.array([170, lower_hsv[1], lower_hsv[2]])
        self.upper_hsv2 = np.array([180, upper_hsv[1], upper_hsv[2]])

    def detect(self, frame, roi):
        x_start, y_start, x_end, y_end = roi
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Melhorias na máscara
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(mask, kernel_large, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]

        if filtered_contours:
            combined_contour = np.vstack(filtered_contours)
            hull = cv2.convexHull(combined_contour)

            area = cv2.contourArea(hull)
            if area > 300:
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # Calcular os momentos do contorno convexo
                moments = cv2.moments(hull)
                if moments["m00"] > 0:  # Verifica se o contorno é válido
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    # Calcular o eixo principal usando os momentos
                    mu20 = moments["mu20"] / moments["m00"]
                    mu02 = moments["mu02"] / moments["m00"]
                    mu11 = moments["mu11"] / moments["m00"]
                    orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                    angle = np.degrees(orientation)

                    # Ajustar o ângulo para a faixa esperada
                    if angle < 0:
                        angle += 180

                    return box, (cx, cy, angle), mask

        return None, None, mask
