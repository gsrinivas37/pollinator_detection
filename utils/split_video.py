"""
This script splits a video into frames at the specified frame rate, resizes these
to specified dimensions, and saves these to the specified directory.
"""

import os
import cv2

def split_video(vidpath, output_dir, desired_fps=4, width=640, height=640):
    vcdata = cv2.VideoCapture(vidpath)
    num_frames = vcdata.get(cv2.CAP_PROP_FRAME_COUNT)
    actual_fps = vcdata.get(cv2.CAP_PROP_FPS)

    # calculate duration of the video
    seconds = num_frames / actual_fps
    desired_frames = int(seconds * desired_fps)
    delta = num_frames / desired_frames

    (W, H) = (None, None)

    # Read in all the frames and store in an array
    frame_num = 0
    frame_array = [None] * desired_frames
    next_delta = 0
    count = 0
    while True:
        # read the next frame from the file
        (grabbed, frame) = vcdata.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if frame_num >= next_delta:
            next_delta += delta
            if H != height or W != width:
                frame_array[count] = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
            else:
                frame_array[count] = frame
            count += 1
        frame_num += 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    [_, vidname] = os.path.split(vidpath)
    [root_fn, _] = os.path.splitext(vidname)
    frames = []
    for count, frame in enumerate(frame_array):
        filename = root_fn + "_" + str(count) + ".jpg"
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, frame)
        frames.append(output_path)
    return frames
