video='taipei-bus'
resize_width=416
resize_height=416

python3 models/resize_image.py \
    --image_path_list vdata/${video}/test.txt \
    --resize_width ${resize_width} \
    --resize_height ${resize_height} \
    --num_threads 10
