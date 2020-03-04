video='taipei-bus'
class='car'
gt_path='gt0.5'
path=${gt_path}'_t0dt0.1'
k=100
conf=0.9
gap=1
calibrate=0

python3 plot.py \
    --image_path_list vdata/${video}/test.txt \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --cm_config_path cfg/yolov3-tiny-${class}.cfg \
    --cm_weight_path vdata/${video}/checkpoint/yolov3-tiny-${class}_final.weights \
    --cm_meta_path cfg/${class}.data \
    --label_path vdata/${video}/${gt_path}/labels.json \
    --class_name ${class} \
    --table_path vdata/${video}/${path}/cptable.json \
    --cm_result_path vdata/${video}/${path}/cm_result.json \
    --calibrator_path vdata/${video}/${gt_path}/calibrator.pkl \
    --log_path vdata/${video}/logs/${path}_k${k}_conf${conf}_gap${gap}_calibrate${calibrate}.log \
    --k ${k} \
    --confidence ${conf} \
    --gap ${gap} \
    --threshold 0 \
    --num_threads 10 \
    --calibrate ${calibrate} \
    --read_cm_result
