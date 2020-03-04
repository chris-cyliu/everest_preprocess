video='taipei-bus'
class='car'

python3 -m models.analyse_model \
    --image_path_list vdata/${video}/val.txt \
    --cm_config_path cfg/yolov3-tiny-${class}.cfg \
    --cm_weight_path vdata/${video}/checkpoint/yolov3-tiny-${class}_final.weights \
    --cm_meta_path cfg/${class}.data \
    --calibrator_path vdata/${video}/gt0.5/calibrator.pkl \
    --resize_width 416 \
    --resize_height 416 \
    --num_threads 10
