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

OPTS=''

if [ ${NUM_FRAME} == "0" ]; then
  /usr/bin/ffmpeg -i ${FILE_VIDEO} ${OPTS} ${DIR_OUTPUT}/%d.jpg
else
  /usr/bin/ffmpeg -i ${FILE_VIDEO} ${OPTS} -vframes ${NUM_FRAME} ${DIR_OUTPUT}/%d.jpg
fi
