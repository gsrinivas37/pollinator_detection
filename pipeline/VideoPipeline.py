import os.path
from utils.split_video import split_video
import shutil
from inference.Yolov7Detector import YoloV7Detector
from utils.yolo_classify_videos import classify_video

DEBUG = True

class VideoPipeline:
    def __init__(self, classifier, detector, fps=3, tmp_dir='tmp', threshold=0.7):
        self.classifier = classifier
        self.detector = detector
        self.fps = fps
        self.threshold = threshold
        self.tmp_dir = tmp_dir
        self.frames = []

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

    def pollinator_exist(self):
        max_cnt = 3
        current_cnt = 0
        for frame in self.frames:
            result = self.classifier.run(frame)
            if result < 5:
                current_cnt += 1
                # If classifier found pollinator for three times, skip remaining and return True
                if current_cnt == max_cnt:
                    if DEBUG:
                        print(f"Classifier found pollinator: {result}")
                    return True
        return False

    def generate_frames(self, video):
        frames_directory = os.path.join(self.tmp_dir, os.path.split(video)[1])
        self.frames = split_video(video, frames_directory)

    def is_pollinating(self, bounding_boxes):
        pollinator = None
        # TODO: Compare bounding boxes of flower and pollinator and return True if they overlap

        return pollinator

    def run(self, video):
        self.generate_frames(video)

        if not self.pollinator_exist():
            if DEBUG:
                print(f"Skipping the video {video} as classifier found no pollinators.")
            return

        if isinstance(self.detector, YoloV7Detector):
            frames_directory = os.path.join(self.tmp_dir, os.path.split(video)[1])
            results_directory = frames_directory+"_results"
            # Run YoloV7 detector on all frames of video as running each frame individually is slow.
            self.detector.run(frames_directory, result_dir=results_directory, remove_result_dir=True)
            labels_directory = os.path.join(results_directory, 'exp', 'labels')
            csv = classify_video(labels_directory)
            print(csv)
            return csv
        else:
            # TODO: Need more work here for EfficientDet flow...
            pollinators = []
            for frame in self.frames:
                if DEBUG:
                    print(f'Running detection on {frame}')
                bounding_boxes = self.detector.run(frame)
                if DEBUG:
                    print(f"Detection are : {bounding_boxes}")

                result = self.is_pollinating(bounding_boxes)
                if result is not None:
                    pollinators.append(result)

            return pollinators
