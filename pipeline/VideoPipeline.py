
class VideoPipeline:
    def __init__(self, classifier, detector, fps=3, threshold=0.7):
        self.classifier = classifier
        self.detector = detector
        self.fps = fps
        self.threshold = threshold
        self.tmp_dir = "tmp"
        self.frames = []

    def pollinator_exist(self):
        max_cnt = 3
        current_cnt = 0
        for frame in self.frames:
            result = self.classifier.run(frame)
            if result:
                current_cnt += 1
                if current_cnt == max_cnt:
                    return True
        return False

    def generate_frames(self, video):
        # TODO: Populate self.frames
        pass

    def is_pollinating(self, bounding_boxes):
        pollinator = None
        # TODO: Compare bounding boxes of flower and pollinator and return True if they overlap

        return pollinator

    def run(self, video):
        self.generate_frames(video)

        if not self.pollinator_exist():
            return

        pollinators = []
        for frame in self.frames:
            bounding_boxes = self.detector.run(frame)
            result = self.is_pollinating(bounding_boxes)
            if result is not None:
                pollinators.append(result)

        return pollinators
