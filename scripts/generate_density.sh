python3 -m models.generate_density \
    --image_path vdata/taipei-bus/train.txt \
    --resize_width 416 \
    --resize_height 416

python3 -m models.generate_density \
    --image_path vdata/taipei-bus/val.txt \
    --resize_width 416 \
    --resize_height 416
