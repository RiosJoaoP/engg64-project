class Detector:
    def __init__(self):
        pass

    def detect(self, frame, roi=None):
        """
        Método principal para detectar objetos em um frame.
        Deve ser implementado pelas subclasses.

        Args:
            frame: O frame da imagem.
            roi: Região de interesse (opcional).

        Returns:
            box: Coordenadas do retângulo delimitador (se encontrado).
            rect: Retângulo rotacionado (se encontrado).
            mask: Máscara binária usada na detecção.
        """
        raise NotImplementedError("O método detect() deve ser implementado pela subclasse.")