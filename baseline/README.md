# Baseline 

Framework: **Tensorflow Yolov3**. 

NMS parameters: score_threshold=0.3, max_boxes=200, IOU_threshold=0.45

## The ground truth by Yolov3 

The path: `~/workspace/data/${VIDEO_NAME}/${VIDEO_NAME}_train.csv`

The file format: 

 |frame |label| x-min | y-min | x-max | y-max | label | x-min | y-min | x-max | y-max |
 |---|----| ------------| -------| ------- | ----- | -------- |------|-----|-----|----|
  | 1| 1 | 0.8         | 0.12 | 0.22 | 0.30  | 0.50 |1  |219.0 |305.4 |129.0 | 135.4|
    
**Please note that the frames are counted from 1**

## The frames

The frames are stored in the path of `~/workspace/data/${VIDEO_NAME}/images/%d.jpg` like,

1.jpg, 2.jpg, 10245.jpg, 10246.jpg, 10257.jpg...

**Please note that the images are strated from 1.jpg (not 0.jpg)** 


