#! /usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "One argument are required: [video_file] [output_dir] [GPU ID]"
    exit 1
fi

BASEPATH=`dirname $0`
FILE_VIDEO=${1}
DIR_OUTPUT=${2}
GPUID=${3}

# Define output dir
DIR_OUTPUT_IMAGES=${DIR_OUTPUT}/images
DIR_OUTPUT_IMAGES_RESIZE=${DIR_OUTPUT}/images_resize
DIR_OUTPUT_LABELS=${DIR_OUTPUT}/labels

TIMESTAMP=$(date +%s)

TMP_FILE_IMAGE_LIST=/tmp/image_list_${TIMESTAMP}.txt
NUM_CORE=$(nproc)

mkdir -p ${DIR_OUTPUT_IMAGES} ${DIR_OUTPUT_IMAGES_RESIZE} ${DIR_OUTPUT_LABELS}

set -x 

sh ${BASEPATH}/scripts/extract_frames.sh ${FILE_VIDEO} ${DIR_OUTPUT_IMAGES}

CUDA_VISIBLE_DEVICES=${3} python ./baseline_inference.py --image_folder=./vdata/${1}/images/ --label_folder=./vdata/${1}/labels/ --time_file=./${1}_time.txt

