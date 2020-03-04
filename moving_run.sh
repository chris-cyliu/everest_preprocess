
sh ./scripts/extract_frames.sh ./vdata/moving moving

ls vdata/moving/images/ | sed "s:^:vdata/moving/images/:" > moving_images

python ./models/resize_image.py --image_path_list=./moving_images --resize_width=416 --resize_height=416 --num_threads=16

CUDA_VISIBLE_DEVICES=2 python ./baseline_inference.py --image_folder=./vdata/moving/images_resize/ --label_folder=./vdata/moving/labels/ --time_file=./moving_time.txt

