import wget
import gdown

YOLO_MODEL = '1_UA7aXN28TcnSALi5MwL1bDThvt7oS0Q'
CLASSIFIER_MODEL = '1sF3U9T2c8O54SiTUcHL9_acZST2pxaZJ'
def download_gdrive_file(FILE_ID, output):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, output, quiet=False)

download_gdrive_file(YOLO_MODEL, "yolov7.pt")
download_gdrive_file(CLASSIFIER_MODEL, "classifier_model.pt")
