#!/bin/sh
DATASET_MAIN_PATH="/home/apsisdev/ansary/DATASETS/APSIS/Detection/"
#DATASET_MAIN_PATH="/home/ansary/WORK/Work/APSIS/datasets/Detection/"
PROCESSED_PATH="${DATASET_MAIN_PATH}processed/"
BASE_DATA_PATH="${DATASET_MAIN_PATH}source/base/"

# synthetic
python synth.py $BASE_DATA_PATH $PROCESSED_PATH --n_data 5000
python store.py "${PROCESSED_PATH}dbsynth/image/" $DATASET_MAIN_PATH dbsynth  


echo succeded