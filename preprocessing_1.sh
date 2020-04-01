#! /usr/bin/env bash
if [ "$#" -ne 5 ]; then
    echo "One argument are required: [video_file] [output_dir] [GPU ID] [SKIP_RESIZE(0|1)] [TARGET_FRAME]"
    exit 1
fi

# enable conda
source ~/.bashrc

BASEPATH=`dirname $0`
FILE_VIDEO=${1}
DIR_OUTPUT=${2}
GPUID=${3}
SKIP_RESIZE=${4}
TARGET_FRAME=${5}

# Define output dir
DIR_OUTPUT_IMAGES=${DIR_OUTPUT}/images
DIR_OUTPUT_IMAGES_RESIZE=${DIR_OUTPUT}/images_resize
DIR_OUTPUT_LABELS=${DIR_OUTPUT}/labels

TIMESTAMP=$(date +%s)

TMP_FILE_IMAGE_LIST=/tmp/image_list_${TIMESTAMP}.txt
NUM_CORE=$(nproc)

#Constant
RESIZE_HEIGHT=416
RESIZE_WIDTH=416

mkdir -p ${DIR_OUTPUT_IMAGES}
export PYTHONPATH=${BASEPATH}
set -x

echo "${1}"

sh ${BASEPATH}/scripts/extract_frames.sh ${FILE_VIDEO} ${DIR_OUTPUT_IMAGES} ${TARGET_FRAME}

#conda activate va_resize
#find ${DIR_OUTPUT_IMAGES} -type f | xargs -P ${NUM_CORE} -n 10000 magick mogrify -resize ${RESIZE_HEIGHT}x${RESIZE_WIDTH}!
find ${DIR_OUTPUT_IMAGES} -type f > ${TMP_FILE_IMAGE_LIST}

mkdir -p ${DIR_OUTPUT_IMAGES_RESIZE}
python3 ${BASEPATH}/models/resize_image.py --image_path_list=${TMP_FILE_IMAGE_LIST} \
--resize_width=${RESIZE_HEIGHT} \
--resize_height=${RESIZE_WIDTH} \
--num_threads=10

# Filter out some error image and delete it
#echo "Drop images with problem (size=0)"
#find ${DIR_OUTPUT_IMAGES} -printf -size 0 -exec rm -f {} \;
#echo "Drop images with problem (contain ~)"
#find ${DIR_OUTPUT_IMAGES} -printf -name '*~' -exec rm -f {} \;
rm -rf  ${DIR_OUTPUT_IMAGES}
mv ${DIR_OUTPUT_IMAGES_RESIZE} ${DIR_OUTPUT_IMAGES}
mkdir ${DIR_OUTPUT_LABELS}

#conda activate base
CUDA_VISIBLE_DEVICES=${GPUID} python ${BASEPATH}/baseline_inference.py \
--image_folder=${DIR_OUTPUT_IMAGES} \
--label_folder=${DIR_OUTPUT_LABELS} \
--time_file=${DIR_OUTPUT}/time.txt

#Clean up intermediate photo
