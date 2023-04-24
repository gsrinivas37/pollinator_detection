import os
import sys

from inference.Classifier import Classifier
from inference.EfficientDetector import EfficientDetector
from inference.Yolov7Detector import YoloV7Detector
from pipeline.VideoPipeline import VideoPipeline


def run_pollinator_detection(input_dir, classifier_type='resnet', detector_type='yolo', output_file='results.csv'):
    model_path = None

    if classifier_type == 'resnet':
        model_path = 'models/resnet_model.pt'
    if classifier_type == 'vgg':
        model_path = 'models/vgg_model.pt'

    detector = None

    if detector_type == 'yolo':
        detector = YoloV7Detector('models/yolov7.pt', 'models/yolov7')
    if detector_type == 'efficient_det':
        detector = EfficientDetector('models/saved_model')

    classifier = Classifier(model_path, classifier_type)
    pipeline = VideoPipeline(classifier, detector)

    results = ""
    for video in os.listdir(input_dir):
        if not video.endswith('.mp4'):
            continue
        print(f'Processing.. {video}')
        results += pipeline.run(os.path.join(input_dir, video))

    with open(output_file, "w") as fd:
        fd.write(results)


run_pollinator_detection('/Users/gsrinivas37/work/test_video')

# if __name__ == "__main__":
#     if len(sys.argv) != 6:
#         print(f"Call is: {sys.argv[0]} <input dir> <classifier_type> <detector_type> <output_file>")
#         sys.exit(1)
#
#     [_, input_dir, classifier_type, detector_type, output_file] = sys.argv
