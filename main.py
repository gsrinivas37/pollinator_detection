import os

from models.Classifier import Classifier
from models.Detector import Detector
from pipeline.VideoPipeline import VideoPipeline

classifier = None
detector = None

SOURCE_DIRECTORY = None

pipeline = VideoPipeline(classifier, detector)

for video in os.listdir(SOURCE_DIRECTORY):
    print(f'Processing.. {video}')
    pollinators = pipeline.run(video)
    #TODO: How should we return results..
