#!/bin/sh
DATASET_MAIN_PATH="/home/apsisdev/ansary/DATASETS/APSIS/Detection/"
#DATASET_MAIN_PATH="/home/ansary/WORK/Work/APSIS/datasets/Detection/"
PROCESSED_PATH="${DATASET_MAIN_PATH}processed/"
BASE_DATA_PATH="${DATASET_MAIN_PATH}source/base/"
SYNTH_TEXT_PATH="/backup/RAW/DET/DATA/synth/temp/image/"
# synthetic
python synth.py $BASE_DATA_PATH $PROCESSED_PATH --n_data 10000
python store.py "${PROCESSED_PATH}dbsynth/image/" $DATASET_MAIN_PATH dbsynth  
# synth text
python store.py $SYNTH_TEXT_PATH "/backup/RAW/DET/DATA/synth/temp/"  synthtext 
echo succeded