# video-analytics

## Implementation design
- config: configure for query, including training parameters for cheap model
- models: definition of cheap models
    - builder.py: definition of cheap model's symbol
- operators: operator implementation for each query
    - op.py: base class of all operators
    - topk.py: top-k operator
- utils: utilities
    - data_sampler.py: sample image data and divide into train, val and test set
    - label_generator.py: generate label from full model
- scripts: scripts
- core: common part of all types of queries
    - engine.py: execute query inside
    - io: read and write videos and images
        - video_io.py: read video and save images to binary file
        - record_io.py: read binary file containing images

## Flow of execution
```
# split video into images
bash scripts/extract_frames.sh vdata/taipei-bus taipei-bus

# sanity check
bash scripts/sanity_check.sh

# split dataset
bash scripts/split_dataset.sh

# generate label for cheap model and top-k
bash scripts/generate_label.sh

# resize image
bash scripts/resize_image.sh

# train cheap model
cd darknet
./darknet detector train cfg/car.data cfg/yolov3-tiny-car.cfg yolov3-tiny.conv.15
./darknet detector map cfg/car.data cfg/yolov3-tiny-car.cfg vdata/taipei-bus/checkpoint/yolov3-tiny-car_final.weights

# run baseline
bash scripts/baseline.sh

# run top-k
bash scripts/run.sh

# plot figures
bash scripts/plot.sh
```

## Tunable parameters
- threshold in difference detector
- IoU threshold in calibrator
- confidence threshold in cheap detector
