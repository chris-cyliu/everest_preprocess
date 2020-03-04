video='taipei-bus'
class='car'
resize_width=416
resize_height=416

python3 -m models.generate_label \
    --image_path_list vdata/${video}/train.txt \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --class_name ${class} \
    --save_path vdata/${video}/labels \
    --resize_width ${resize_width} \
    --resize_height ${resize_height} \
    --sample_size 5000 \
    --num_threads 10 \
    --type bbox

python3 -m models.generate_label \
    --image_path_list vdata/${video}/val.txt \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --class_name ${class} \
    --save_path vdata/${video}/labels \
    --resize_width ${resize_width} \
    --resize_height ${resize_height} \
    --sample_size 2000 \
    --num_threads 10 \
    --type bbox

python3 -m models.generate_label \
    --image_path_list vdata/${video}/test.txt \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --class_name ${class} \
    --save_path vdata/${video}/labels.json \
    --resize_width ${resize_width} \
    --resize_height ${resize_height} \
    --num_threads 10 \
    --type count
