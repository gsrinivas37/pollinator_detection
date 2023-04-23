class Detector:
    def __int__(self, model_path, repo_location, type='yolo'):
        self.type = type
        self.model_path = model_path
        self.repo_location = repo_location

    def run(self, image):
        """
        Args:
            image: Path to image file

        Returns:

        """
        return None
