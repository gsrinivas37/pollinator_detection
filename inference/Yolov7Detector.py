import os.path

from inference.Detector import Detector
import subprocess
import shutil
from shared import run_command


class YoloV7Detector(Detector):
    def __init__(self, model_path, repo_path, threshold=0.7):
        super().__init__(model_path, threshold)
        self.repo_path = repo_path

    def run(self, image, result_dir='runs', remove_result_dir=False, img_size=640, nosave=True):
        # Clear the directory if it exists
        if remove_result_dir and os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        command = f"python {os.path.join(self.repo_path, 'detect.py')} --weights {self.model_path} " \
              f"--img-size {img_size} --source {image} --save-txt --project {result_dir}"
        if nosave:
            command += " --nosave"

        print(run_command(command))


if __name__ == "__main__":
    image_path = '/Users/gsrinivas37/work/test_img'

    detector = YoloV7Detector('../models/yolov7.pt', "../models/yolov7")
    detector.run(image_path, result_dir='results')

