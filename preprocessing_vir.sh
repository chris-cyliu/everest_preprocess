#! /usr/bin/env bash

if [ "$#" -ne 3 ]; then 
    echo "One argument are required: [video_path] [video_name] [GPU ID]"
    exit 1
fi

set -x 

echo "${1}"

mkdir -p vdata/${1}/images
mkdir -p vdata/${1}/labels

sh ./scripts/extract_frames_1.sh ./vdata/${1} ${2}

CUDA_VISIBLE_DEVICES=${3} python ./baseline_inference.py --image_folder=./vdata/${1}/images/ --label_folder=./vdata/${1}/labels/ --time_file=./${1}_time.txt

