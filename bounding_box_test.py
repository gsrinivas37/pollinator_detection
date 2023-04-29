import os

import matplotlib.pyplot as plt
import numpy as np

from shared import *

labels = ["Bees-Wasps",
          "Bumble Bee",
          "Butterflies-Moths",
          "Fly",
          "Hummingbird",
          "Inflorescence",
          "Other"]

def get_bbox_stats(dataset):
    img_ids = dataset.coco.getImgIds()
    sizes = dict()
    for img_id in img_ids:
        # Read in image and obtain its annotations
        img_info = dataset.coco.loadImgs(img_id)[0]
        img_annots = dataset.coco.imgToAnns[img_id]
        image_path = os.path.join(dataset.root, img_info['file_name'])

        img = cv2.imread(image_path)

        if len(img_annots) == 0:
            continue

        detections = sv.Detections.from_coco_annotations(coco_annotation=img_annots)
        h, w, _ = img.shape
        for box, _, _, class_id, _ in detections:
            width = int(box[2] - box[0])
            height = int(box[3] - box[1])
            size = (min(height, width) / h)
            if class_id not in sizes:
                sizes[class_id] = []
            sizes[class_id].append(size)
            # if class_id == 2:
            #     if not os.path.exists('test_imgs'):
            #         os.mkdir('test_imgs')
            #     run_command(f'cp {image_path} test_imgs/')
            #     print('copy fie...')
    return sizes


root_dir = '/Users/gsrinivas37/Old_Download/Pollinators-18_COCO_640x640_aug_null_nf/train'
dataset = load_dataset(root_dir)
sizes = get_bbox_stats(dataset)
for i in sizes:
    values = sizes[i]
    plt.clf()
    plt.title(f"{labels[i-1]} [{i}]: Avg: {np.round_(np.mean(np.array(values)),3)}")
    plt.hist(np.array(values), range=[0,0.3])
    plt.savefig(f'class_{i}.png')
