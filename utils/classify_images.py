import os
import sys
import pandas as pd

def get_video_base_names(labels_dir):
    """Returns a sorted list of all the video base names associated with the label files
    in the specified directory. It assumes yolo formated label files have names of the
    following form:
        <video base name>_<count>.txt
    """
    base_names_set = set()
    for fn in os.listdir(labels_dir):
        if fn[-4:] != ".txt":
            continue
        if '_' not in fn:
            continue
        vidbase = fn[:fn.rfind('_')]
        base_names_set.add(vidbase)
    base_names = list(base_names_set)
    base_names.sort()
    return base_names

def get_label_filenames_for_video(labels_dir, vid_base_name):
    filenames = []
    for fn in os.listdir(labels_dir):
        if fn.find(vid_base_name) == 0:
            filenames.append(fn)
    filenames.sort()
    return filenames

def get_label_filenames_for_images(labels_dir):
    filenames = []
    for fn in os.listdir(labels_dir):
            filenames.append(fn)
    filenames.sort()
    return filenames

def classify_images(labels_dir: str, images_dir: str, classifier_1, classifier_2):
    if not os.path.exists(labels_dir):
        print(f"ERROR: {labels_dir} does not exit")
        sys.exit(2)

    vid_base_names = get_video_base_names(labels_dir)
    csv = ""

    for vbn in vid_base_names:
        csv_line = vbn

        # Get all the label file names for this video
        fns = get_label_filenames_for_video(labels_dir, vbn)

        label_df = pd.DataFrame(columns=['image_fn', 'label_fn', 'detector_out'])
        label_df = label_df.set_index(['image_fn'])

        for fn in fns:
            fp = os.path.join(labels_dir, fn)
            image_fn = fn.strip('.txt') + '.jpg'
            with open(fp, "r") as fd:
                for line in fd:
                    line = line.strip()
                    line = line.rstrip()
                    if len(line) < 6:
                        continue
                    label_info = line.split(' ')
                    if len(label_info) < 5:
                        continue

                    cls = int(label_info[0])

                    if cls != 5:
                        label_df.loc[image_fn] = [fn, cls]

        if label_df.empty:
            continue

        label_df = run_classifier(images_dir, label_df, classifier_1, classifier_2)
        changed_label_df = label_df[label_df['final_out'] != label_df['detector_out']]  ## New Ensemble labels
        print('Ensemble output labels to be overwritten: ', len(changed_label_df))


        for i in range(len(changed_label_df)):
            fn = changed_label_df.iloc[i]['label_fn']
            overwrite = False
            with open(os.path.join(labels_dir, fn), "r+") as fd:
                lines = []

                for line in fd:
                    line = line.strip()
                    line = line.rstrip()
                    if len(line) < 6:
                        continue
                    label_info = line.split(' ')
                    if len(label_info) < 5:
                        continue

                    cls = int(label_info[0])

                    if cls != 5:
                        label_info[i] = str(changed_label_df.iloc[i]['final_out'])
                        overwrite = True
                    lines.append(' '.join(label_info))

                if overwrite:
                    fd.seek(0)
                    fd.truncate()
                    fd.write('\n'.join(lines))



def run_classifier(images_dir, label_df, classifier_1, classifier_2):

    label_df['classifier1_out'] = classifier_1.run(images_dir, list(label_df.index))

    label_df['classifier2_out'] = classifier_2.run(images_dir, list(label_df.index))

    label_df['final_out'] = label_df[['detector_out', 'classifier1_out', 'classifier2_out']].mode(axis=1)

    return label_df