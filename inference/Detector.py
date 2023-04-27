class Detector:
    def __init__(self, model_path, device, threshold=0.7):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device

    def run(self, image):
        pass
