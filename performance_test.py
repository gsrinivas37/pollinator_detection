from inference.Classifier import Classifier
from inference.EfficientDetector import EfficientDetector
from inference.Yolov7Detector import YoloV7Detector
import time

image_path = '/Users/gsrinivas37/work/test_img/one.jpg'

start_time = time.time()
vgg_classifier = Classifier('models/vgg_model.pt', 'vgg')
end_time = time.time()
print(f"VGG Model loaded in {end_time - start_time} seconds")

start_time = time.time()
resnet_classifier = Classifier('models/resnet_model.pt', 'resnet')
end_time = time.time()
print(f"ResNet Model loaded in {end_time - start_time} seconds")

start_time = time.time()
vgg_classifier.run(image_path)
end_time = time.time()
print(f"VGG Model predicted in {end_time - start_time} seconds")

start_time = time.time()
resnet_classifier.run(image_path)
end_time = time.time()
print(f"ResNet Model predicted in {end_time - start_time} seconds")

detector = EfficientDetector('models/saved_model')

start_time = time.time()
detector.run(image_path)
end_time = time.time()
print(f"EfficientDet Model predicted in {end_time - start_time} seconds")

start_time = time.time()
yolo_detector = YoloV7Detector('models/yolov7.pt', 'yolov7')
# print(yolo_detector.run('tmp'))  # Check running on a folder with multiple images
print(yolo_detector.run(image_path))
end_time = time.time()
print(f"Yolo Detector Model loaded and predicted in {end_time - start_time} seconds")