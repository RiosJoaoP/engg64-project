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
        """
        Detecta objetos vermelhos em uma região de interesse (ROI).
        
        :param frame: Quadro de vídeo atual.
        :param roi: Região de interesse (x_início, y_início, x_fim, y_fim).
        :return: Caixa delimitadora, retângulo rotacionado, e máscara.
        """
        # Aplica a ROI
        x_start, y_start, x_end, y_end = roi
        cropped_frame = frame[y_start:y_end, x_start:x_end]

        # Converte para espaço de cor HSV
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Cria máscaras para a cor vermelha
        mask1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Redução de ruído
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Operações morfológicas para refinar a máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontra contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Encontra o maior contorno
            largest_contour = max(contours, key=cv2.contourArea)

            # Calcula área e perímetro para filtrar objetos
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if area > 300 and perimeter > 50:
                # Obtém retângulo rotacionado
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                return box, rect, mask

        return None, None, mask
