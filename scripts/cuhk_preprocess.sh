#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
	echo "Two arguments are required: [VIDEO_NAME] [TARGE_NUM_FRAME]"
	exit 1
fi

VIDEO_NAME=$1
TARGE_NUM_FRAME=$2
BASEPATH=`dirname $0`/..
REMOTE_SRC=pc88161:/media/4tb/video_analytic/raw_video
LOCAL_DST=/data/ssd/public/cyliu/video_analytic/preprocess
REMOTE_DST=pc88161:/media/4tb/video_analytic/preprocessed_videos

set -x
#Download video
mkdir -p ${LOCAL_DST}/raw_video
scp ${REMOTE_SRC}/${VIDEO_NAME}.mp4 ${LOCAL_DST}/raw_video

VIDEO_PATH=${LOCAL_DST}/raw_video/${VIDEO_NAME}.mp4

${BASEPATH}/preprocessing_1.sh ${VIDEO_PATH} ${LOCAL_DST}/${VIDEO_NAME} 0 0 ${TARGE_NUM_FRAME}

cd ${LOCAL_DST}/${VIDEO_NAME}
find * -type f | tar -zcvf ${LOCAL_DST}/${VIDEO_NAME}.tar.gz -T -

#SCP back
scp ${LOCAL_DST}/${VIDEO_NAME}.tar.gz ${REMOTE_DST}/${VIDEO_NAME}.tar.gz
