#! /usr/bin/env bash

if [ "$#" -ne 2 ]; then 
    echo "One argument are required: [video_name] [GPU ID]"
    exit 1
fi

set -x 

echo "${1}"

mkdir -p vdata/${1}/images
mkdir -p vdata/${1}/images_resize
mkdir -p vdata/${1}/labels

sh ./scripts/extract_frames.sh ./vdata/${1} ${1}

ls vdata/${1}/images/ | sed "s:^:vdata/${1}/images/:" > ${1}_images

python ./models/resize_image.py --image_path_list=./${1}_images --resize_width=416 --resize_height=416 --num_threads=16

CUDA_VISIBLE_DEVICES=${2} python ./baseline_inference.py --image_folder=./vdata/${1}/images_resize/ --label_folder=./vdata/${1}/labels/ --time_file=./${1}_time.txt

