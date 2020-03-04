

sh ./scripts/extract_frames_1.sh vdata/vir_50 vehicles_comp_50


CUDA_VISIBLE_DEVICES=3 python ./baseline_inference.py --image_folder=./vdata/vir_50/images/ --label_folder=./vdata/vir_50/labels/ --time_file=./vir_50_time.txt

