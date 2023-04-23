class Classifier:
    def __init__(self, model_path, type):
        self.model_path = model_path
        self.type = type  # ResNet or VGG

    def run(self, image):
        """
        Args:
            image: Path to the image file

        Returns:
            Whether pollinator exists in the image or not
        """
        return True
