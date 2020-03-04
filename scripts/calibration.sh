video='taipei-bus'
class='car'

python3 -m models.calibration \
    --image_path_list vdata/${video}/val.txt \
    --save_path vdata/${video}/calibrator.pkl \
    --cm_config_path cfg/yolov3-tiny-${class}.cfg \
    --cm_weight_path vdata/${video}/checkpoint/yolov3-tiny-${class}_final.weights \
    --cm_meta_path cfg/${class}.data \
    --resize_width 416 \
    --resize_height 416 \
    --iou_thr 0 \
    --num_threads 10
