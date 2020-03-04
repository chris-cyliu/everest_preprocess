
sh ./scripts/extract_frames.sh ./vdata/thailand-bay thailand-bay 

ls vdata/thailand-bay/images/ | sed "s:^:vdata/thailand-bay/images/:" > thailand-bay_images

python ./models/resize_image.py --image_path_list=./thailand-bay_images --resize_width=416 --resize_height=416 --num_threads=8

CUDA_VISIBLE_DEVICES=0 python ./baseline_inference.py --image_folder=./vdata/thailand-bay/images_resize/ --label_folder=./vdata/thailand-bay/labels/ --time_file=./thailand-bay_time.txt

