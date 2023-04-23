import wget
import gdown
import zipfile

YOLO_MODEL = '1_UA7aXN28TcnSALi5MwL1bDThvt7oS0Q'
EFFICIENTDET_MODEL = '1tWAozTBGQ-Xw_oIhqhH6scGR3k9yNMNN'
VGG_MODEL = '1sF3U9T2c8O54SiTUcHL9_acZST2pxaZJ'
RESNET_MODEL = '1YIPPXg0YfGsxHy-ZiilKDXoISbITmhve'

def download_gdrive_file(FILE_ID, output):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, output, quiet=False)
    if output.endswith('.zip'):
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(".")


# download_gdrive_file(YOLO_MODEL, "models/yolov7.pt")
#download_gdrive_file(EFFICIENTDET_MODEL, 'models/saved_model.zip')
download_gdrive_file(VGG_MODEL, "../models/vgg_model.pt")
download_gdrive_file(RESNET_MODEL, "../models/resnet_model.pt")