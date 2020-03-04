video='taipei-bus'
image='5469448.jpg'
class='car'

python3 -m models.visualize \
    --image_path vdata/${video}/images/${image} \
    --config_path cfg/yolov3.cfg \
    --weight_path yolov3.weights \
    --meta_path cfg/coco.data \
    --calibrator_path vdata/${video}/gt0.5/calibrator.pkl \
    --class_name ${class} \
    --resize_width 416 \
    --resize_height 416 \
    --threshold 0.5

#python3 -m models.visualize \
#    --image_path vdata/${video}/images/${image} \
#    --config_path cfg/yolov3-tiny-${class}.cfg \
#    --weight_path vdata/${video}/checkpoint/yolov3-tiny-${class}_final.weights \
#    --meta_path cfg/${class}.data \
#    --calibrator_path vdata/${video}/gt0.5/calibrator.pkl \
#    --class_name ${class} \
#    --resize_width 416 \
#    --resize_height 416 \
#    --threshold 0.1 \
#    --calibrate
