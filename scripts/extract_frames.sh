#! /usr/bin/env bash

if [ "$#" -ne 2 ]; then 
	echo "Two arguments are required: [video_dir] [video_name]"
	exit 1
fi 

set -x 

if [ ! -d "${1}" ]; then
    mkdir -p ${1}
fi

ffmpeg -i ${1}/${2}.mp4 ${1}/images/%d.jpg 
