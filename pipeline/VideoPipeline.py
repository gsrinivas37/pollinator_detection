import os.path
from utils.split_video import split_video
import shutil
from inference.Yolov7Detector import YoloV7Detector
from utils.yolo_classify_videos import classify_video
from utils.classify_images import classify_images

class VideoPipeline:
    def __init__(self, classifier_1, classifier_2, detector, fps=3, tmp_dir='tmp', threshold=0.7):
        self.classifier_1 = classifier_1
        self.classifier_2 = classifier_2
        self.detector = detector
        self.fps = fps
        self.threshold = threshold
        self.tmp_dir = tmp_dir
        self.frames = []

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)


    def generate_frames(self, video):
        frames_directory = os.path.join(self.tmp_dir, os.path.split(video)[1])
        self.frames = split_video(video, frames_directory)

    def is_pollinating(self, bounding_boxes):
        pollinator = None
        # TODO: Compare bounding boxes of flower and pollinator and return True if they overlap

        return pollinator

    def run(self, video):
        self.generate_frames(video)


        if isinstance(self.detector, YoloV7Detector):
            frames_directory = os.path.join(self.tmp_dir, os.path.split(video)[1])
            results_directory = frames_directory+"_results"
            # Run YoloV7 detector on all frames of video as running each frame individually is slow.
            self.detector.run(frames_directory, result_dir=results_directory, remove_result_dir=True)
            labels_directory = os.path.join(results_directory, 'exp', 'labels')
            classify_images(labels_dir=labels_directory, images_dir=frames_directory, classifier_1=self.classifier_1, classifier_2=self.classifier_2)
            csv = classify_video(labels_directory)
            print('Yolo output: ', csv)
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
