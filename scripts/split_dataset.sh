video='taipei-bus'

python3 models/split_dataset.py \
    --video_path vdata/${video}/${video}.mp4 \
    --image_dir vdata/${video}/images \
    --save_dir vdata/${video}
