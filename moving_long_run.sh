
sh ./scripts/extract_frames.sh ./vdata/moving_long moving_long

ls vdata/moving_long/images/ | sed "s:^:vdata/moving_long/images/:" > moving_long_images

python ./models/resize_image.py --image_path_list=./moving_long_images --resize_width=416 --resize_height=416 --num_threads=16

CUDA_VISIBLE_DEVICES=1 python ./baseline_inference.py --image_folder=./vdata/moving_long/images_resize/ --label_folder=./vdata/moving_long/labels/ --time_file=./moving_long_time.txt

