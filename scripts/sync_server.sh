#!/bin/env bash

BASEPATH=`dirname $0`/../

cd ${BASEPATH}
rsync -avz --exclude '.git' --exclude 'venv*' * pc88161:video_analytic/code
#rsync -avz --exclude '.git' --exclude 'venv*' * linux9:/uac/rshr/cyliu/bigDataStorage/video_analytic/preprocess