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
        
        # Para lidar com tons de vermelho que cruzam o limite do espectro HSV
        self.lower_hsv2 = np.array([170, lower_hsv[1], lower_hsv[2]])
        self.upper_hsv2 = np.array([180, upper_hsv[1], upper_hsv[2]])

    def detect(self, frame, roi):
        x_start, y_start, x_end, y_end = roi
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Máscaras para vermelho
        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Operações de suavização
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Expandir a máscara para evitar divisões
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.dilate(mask, kernel_large, iterations=2)

        # Combinar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            combined_contour = np.vstack(contours)
            hull = cv2.convexHull(combined_contour)

            # Calcular caixa delimitadora
            area = cv2.contourArea(hull)
            if area > 300:
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                return box, rect, mask

        return None, None, mask

