
# Author: pfzhang (pfzhang@cse.cuhk.edu.hk)

import cv2
import numpy as np
from math import ceil
# This is video iterator.
def VideoIterator(video_path, scale=None, interval=1, start=0):

    is_version4 = cv2.__version__[0] == '4'
    if not is_version4:
        print("Please install OpenCV 4.x.x")
        exit(1)

    cap = cv2.VideoCapture(video_path)

    # Seeks to N-th frame. The next read is the N+1-th frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start-1)
    frame = 0
    frame_ind = -1

    # yield frame_ind and frame
    while frame is not None:
        frame_ind += 1
        _, frame = cap.read()

        if frame_ind % interval != 0:
            continue

        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        yield frame_ind, frame



# Get frames from the video.
def get_frames(num_frames, video_path, scale=None, interval=1, start=0, dtype='float32'):

    true_num_frames = int(ceil((num_frames + 0.0) / interval))
    print('%d total frames / %d frame interval = %d actual frame'
          % (num_frames, interval, true_num_frames))

    vid_it = VideoIterator(video_path=video_path, scale=scale, interval=interval, start=start)

    _, frame = vid_it.next()
    frames = np.zeros(tuple([true_num_frames] + list(frame.shape)), dtype=dtype)
    frames[0,:] = frame

    for i in range(1, true_num_frames):
        _, frame = vid_it.next()
        frame[i, :] = frame

    if dtype == 'float32':
        frames /= 255.0

    return frames


