

#sh ./scripts/extract_frames_1.sh vdata/vir vehicle_com_100


CUDA_VISIBLE_DEVICES=0 python ./baseline_inference.py --image_folder=./vdata/vir/images/ --label_folder=./vdata/vir/labels/ --time_file=./vir_time.txt

