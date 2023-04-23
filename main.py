import os

from models.Classifier import Classifier
from models.Detector import Detector
from pipeline.VideoPipeline import VideoPipeline

classifier = Classifier('classifier_model.pt', 'VGG')
detector = Detector('yolov7.pt', 'yolov7', 'YOLO')

SOURCE_DIRECTORY = '/Users/gsrinivas37/work/test_video'

pipeline = VideoPipeline(classifier, detector)

for video in os.listdir(SOURCE_DIRECTORY):
    print(f'Processing.. {video}')
    pollinators = pipeline.run(os.path.join(SOURCE_DIRECTORY, video))
    # TODO: How should we return results..
