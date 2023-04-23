import os

from models.Classifier import Classifier
from models.Detector import Detector
from models.EfficientDetector import EfficientDetector
from pipeline.VideoPipeline import VideoPipeline

classifier = Classifier('classifier_model.pt', 'VGG')
detector = EfficientDetector('models/saved_model')

SOURCE_DIRECTORY = '/Users/gsrinivas37/work/test_video'

pipeline = VideoPipeline(classifier, detector)

for video in os.listdir(SOURCE_DIRECTORY):
    if not video.endswith('.mp4'):
        continue
    print(f'Processing.. {video}')
    pollinators = pipeline.run(os.path.join(SOURCE_DIRECTORY, video))
    # TODO: How should we return results..
