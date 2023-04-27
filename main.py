import os
import sys

from inference.Classifier import Classifier
from inference.EfficientDetector import EfficientDetector
from inference.Yolov7Detector import YoloV7Detector
from pipeline.VideoPipeline import VideoPipeline


def run_pollinator_detection(input_dir, detector_type='yolo', device='cpu', output_file='results.csv'):
    classifier_type_1 = 'resnet'
    model_path_1 = 'models/resnet_model.pt'
    classifier_type_2 = 'vgg'
    model_path_2 = 'models/vgg_model.pt'

    detector = None

    if detector_type == 'yolo':
        detector_device = device
        if device =='cuda':
            detector_device = 0
        detector = YoloV7Detector('C:/Gaurav/Code/DL/pollinator_detection/models/yolov7.pt', 'C:/Gaurav/Code/DL/pollinator_detection/models/yolov7', device=detector_device)
    if detector_type == 'efficient_det':
        detector = EfficientDetector('models/saved_model')

    classifier_1 = Classifier(model_path_1, classifier_type_1,device=device)
    classifier_2 = Classifier(model_path_2, classifier_type_2,device=device)

    pipeline = VideoPipeline(classifier_1, classifier_2, detector)

    results = ""
    for video in os.listdir(input_dir):
        if not video.endswith('.mp4'):
            continue
        print(f'Processing.. {video}')
        results += pipeline.run(os.path.join(input_dir, video))

    with open(output_file, "w") as fd:
        fd.write(results)


run_pollinator_detection('C:/Gaurav/Code/DL/test video', detector_type='yolo', device='cpu', output_file='results.csv')

# if __name__ == "__main__":
#     if len(sys.argv) != 6:
#         print(f"Call is: {sys.argv[0]} <input dir> <classifier_type> <detector_type> <output_file>")
#         sys.exit(1)
#
#     [_, input_dir, classifier_type, detector_type, output_file] = sys.argv
