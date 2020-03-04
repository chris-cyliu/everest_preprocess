video='taipei-bus'

python3 -m baseline.baseline \
    --image_path_list vdata/${video}/test.txt \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --class_name car \
    --label_path vdata/${video}/gt0.5/labels.json \
    --k 100 \
    --num_threads 10 \
    --read_label
