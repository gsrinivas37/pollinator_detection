import tensorflow as tf
import numpy as np
from PIL import Image
import time
from models.Detector import Detector


class EfficientDetector(Detector):
    def __init__(self, model_path, threshold=0.7):
        super().__init__(model_path, threshold)
        # Load saved model and build the detection function
        print(f"Loading the model {model_path} .... ")
        start_time = time.time()
        self.detect_fn = tf.saved_model.load(model_path)
        end_time = time.time()
        print(f"Model loaded in {end_time-start_time} seconds")

    def run(self, image):
        bounding_boxes = []
        image_np = np.array(Image.open(image))

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        pred_classes = detections['detection_classes']
        pred_scores = detections['detection_scores']
        pred_boxes = detections['detection_boxes']

        for idx in range(len(pred_scores)):
            if pred_scores[idx] > self.threshold:
                bounding_boxes.append((pred_classes[idx], pred_scores[idx], pred_boxes[idx]))
            else:
                break
        return bounding_boxes
