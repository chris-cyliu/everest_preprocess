#! /usr/bin/env bash

if [ "$#" -ne 3 ]; then
	echo "Two arguments are required: [video_file] [output_dir] [num of frame]"
	exit 1
fi

FILE_VIDEO=${1}
DIR_OUTPUT=${2}
NUM_FRAME=${3}


set -x 

if [ ! -d "${DIR_OUTPUT}" ]; then
    mkdir -p ${DIR_OUTPUT}
fi

if [ ${NUM_FRAME} == "0" ]; then
  ffmpeg -i ${FILE_VIDEO} ${DIR_OUTPUT}/%d.jpg
else
  ffmpeg -i ${FILE_VIDEO} -vframes ${NUM_FRAME} ${DIR_OUTPUT}/%d.jpg
fi
