import cv2
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

import tensorflow as tf
import time

import shutil
import json
from copy import copy, deepcopy
import cv2
import supervision as sv
import torchvision

labels = ["Bees-Wasps",
          "Bumble-Bee",
          "Butterflies-Moths",
          "Fly",
          "Hummingbird",
          "Inflorescence", 
          "Other"]

def show_image_with_box(image_path, boxes):
    x = np.array(Image.open(image_path), dtype=np.uint8)
      
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(x)

    # Create a Rectangle patch
    for xmin, ymin, xmax, ymax, cls in boxes:
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                             edgecolor='r', facecolor="none")
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

def get_bbox(file, img_dir, lbl_dir):
    lbl_file = os.path.join(lbl_dir, str(file)[0:-3]+"txt")
    image_path = os.path.join(img_dir, str(file))
    img = cv2.imread(image_path)
    y,x,h = img.shape

    if not os.path.exists(lbl_file):
        print(f"Label doesnt exist for : {lbl_file}")
        return None

    file1 = open(lbl_file, 'r')

    boxes = []
    for line in file1.readlines():
        splits = line.split(" ")
        cls = labels[int(splits[0])]

        center_x  = int(x * float(splits[1]))
        center_y  = int(y * float(splits[2]))
        width  = int(x * float(splits[3]))
        height  = int(y * float(splits[4]))

        xmin = center_x - int(width/2)
        ymin = center_y - int(height/2)
        xmax = center_x + int(width/2)
        ymax = center_y + int(height/2)

        boxes.append((xmin, ymin, xmax, ymax, cls))
    return boxes

def write_to_file(content, file, delete=False):
    if delete and os.path.exists(file):
        os.remove(file)
    if os.path.exists(file):
        mode = 'a' # append if already exists
    else:
        mode = 'w' # make a new file if not

    f = open(file, mode)
    f.write(content+"\n")
    f.close()

def split_data(directory, ratio=0.8, train_csv="train.csv", test_csv="test.csv"):
    img_dir = os.path.join(directory, "images")
    lbl_dir = os.path.join(directory, "labels")

    i = 0
    files = os.listdir(img_dir)
    print(f"Number of files: {len(files)}")
    all_idx = np.arange(len(files))
    np.random.shuffle(all_idx)
    split_idx = int(len(files)*ratio)
    train_idxes = all_idx[:split_idx]
    val_idxes = all_idx[split_idx:]
    write_to_file("filename,width,height,class,xmin,ymin,xmax,ymax", train_csv,True)
    write_to_file("filename,width,height,class,xmin,ymin,xmax,ymax", test_csv,True)
    for idx in train_idxes:
        file = files[idx]
        image_path = os.path.join(img_dir, str(file))
        img = cv2.imread(image_path)
        y,x,h = img.shape

        boxes = get_bbox(file, img_dir, lbl_dir)
        for xmin, ymin, xmax, ymax, cls in boxes:
            write_to_file(f"{file},{x},{y},{cls},{xmin},{ymin},{xmax},{ymax}", train_csv)
    for idx in val_idxes:
            file = files[idx]
            image_path = os.path.join(img_dir, str(file))
            img = cv2.imread(image_path)
            y,x,h = img.shape

            boxes = get_bbox(file, img_dir, lbl_dir)
            for xmin, ymin, xmax, ymax, cls in boxes:
                write_to_file(f"{file},{x},{y},{cls},{xmin},{ymin},{xmax},{ymax}", test_csv)     
    
    
def load_data(directory, print_csv=True, show_image=False, max_cnt=None): 
    img_dir = os.path.join(directory, "images")
    lbl_dir = os.path.join(directory, "labels")

    i = 0
    files = os.listdir(img_dir)
    print(f"Number of files: {len(files)}")
    
    for file in files:
        image_path = os.path.join(img_dir, str(file))
        img = cv2.imread(image_path)
        y,x,h = img.shape

        boxes = get_bbox(file, img_dir, lbl_dir)
        if print_csv:
            for xmin, ymin, xmax, ymax, cls in boxes:
                print(f"{file},{x},{y},{cls},{xmin},{ymin},{xmax},{ymax}")
    
        if show_image:
            show_image_with_box(image_path, boxes)

        i = i + 1
        if max_cnt is not None and i > max_cnt:
            break

def get_model(model_path):
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    model = tf.saved_model.load(model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return model

def get_predicted_label(model, img_path, threshold=0.7):
    label = set()
    image_np = np.array(Image.open(img_path))
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = model(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    pred_classes = detections['detection_classes']
    pred_scores = detections['detection_scores']
    
    for idx in range(len(pred_scores)):
        if pred_scores[idx] > threshold:
            if pred_classes[idx] == 6:
                continue
            label.add(pred_classes[idx])
        else:
            break

    l_list = list(label)
    if len(l_list) == 0:
        return 6
    return l_list[0]

def load_dataset(src_dir):
    annots_path = os.path.join(src_dir, "_annotations.coco.json")
    if not os.path.exists(annots_path):
        print(f"ERROR: No annotations file f{annots_path}")
        return None
    annots = None
    with open(annots_path,"r") as fd:
        annots = json.load(fd)
    if annots is None:
        print(f"ERROR: Reading {annots_path}")    
    dataset = torchvision.datasets.CocoDetection(src_dir, annots_path)
    return dataset

def get_true_label(img_path, dataset=None):
    true_label = set()
    src_dir = os.path.dirname(img_path)
    if dataset is None:
        dataset = load_dataset(src_dir)

    categories =dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}

    img_ids = dataset.coco.getImgIds()
    ann_ids = dataset.coco.getAnnIds()
    
    for img_id in img_ids:
        # Read in image and obtain its annotations
        img_info = dataset.coco.loadImgs(img_id)[0]
        img_annots = dataset.coco.imgToAnns[img_id]
        image_path = os.path.join(dataset.root, img_info['file_name'])
        
        if img_path != image_path:
            continue
        
        img = cv2.imread(image_path)

        if len(img_annots) == 0:
            continue
            
        detections = sv.Detections.from_coco_annotations(coco_annotation=img_annots)
        box_annotator = sv.BoxAnnotator()

        h, w, _ = img.shape
        for box, _, _, class_id, _ in detections:
            if class_id != 6:
                true_label.add(class_id)

    l_list = list(true_label)
    if len(l_list) == 0:
        return 6
    return l_list[0]

def compute_accuracy(root_dir, model_path):
    model = get_model(model_path)
    dataset = load_dataset(root_dir)

    true_labels = []
    pred_labels = []
    for img in os.listdir(root_dir):
        if 'json' in img:
            continue
        img_path = os.path.join(root_dir,img)
        true_labels.append(get_true_label(img_path, dataset))
        pred_labels.append(get_predicted_label(model, img_path))
        
    correct, wrong = dict(), dict()
    for i in range(len(true_labels)):
        t_lbl = true_labels[i]
        if t_lbl == pred_labels[i]:
            if t_lbl in correct:
                correct[t_lbl].append(i)
            else:
                correct[t_lbl] = [i]
        else:
            if t_lbl in wrong:
                wrong[t_lbl].append(i)
            else:
                wrong[t_lbl] = [i]
    all_keys = list(correct.keys()) + list(wrong.keys())
    correct_cnt = len(wrong.values())
    wrong_cnt = len(correct.values())
    print(f"Total Accuracy is: {correct_cnt/(correct_cnt+wrong_cnt)}")
    for class_idx in (sorted(set(all_keys))):
        if class_idx in correct:
            correct_cnt = len(correct[class_idx])
        else:
            correct_cnt = 0

        if class_idx in wrong:
            wrong_cnt = len(wrong[class_idx])
        else:
            wrong_cnt = 0

        print(f"Accuracy for class {class_idx} (Count: {correct_cnt+wrong_cnt}) is: {correct_cnt/(correct_cnt+wrong_cnt)}")