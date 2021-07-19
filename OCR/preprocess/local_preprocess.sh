#!/bin/sh

# Run:
# sudo ./local_preprocess.sh '/media/6TB/alpr_data/experiment'

OUT_DIR=$1

SCALEAI="/media/6TB/alpr_data/scaleAI_all_07072021"
US5="/media/6TB/alpr_data/us5_08312020"
PREV_VALID="/media/6TB/alpr_data/ocr_PLv5_valid_old_and_new"
PREV_TRAIN="/media/6TB/alpr_data/ocr_PLv5_train_old_and_new"

SPLIT_RATE=0.9
MIN_SCORE=0.1
AUG_PER_PLATE_SCALEAI=25
AUG_PER_PLATE_US5=5
STACKED_COEFF=10
SEED=42

LOG_DIR="$OUT_DIR/logging"
EXTRACTED="$OUT_DIR/extracted_scaleai"
DEBUG="$OUT_DIR/errors_scaleai"
EXTRACTED_US5="$OUT_DIR/extracted_us5"
DEBUG_US5="$OUT_DIR/errors_us5"
VALID="$OUT_DIR/valid"
VALID_LMDB="$OUT_DIR/valid_lmdb"
TRAIN="$OUT_DIR/train"
TRAIN_AUGM="$OUT_DIR/augm"
TRAIN_LMDB="$OUT_DIR/augm_lmdb"

PRECESSES=8

python3 extract_plates.py --indir $US5 --outdir $EXTRACTED_US5 --logdir $LOG_DIR --clear_outdir 1 --min_score $MIN_SCORE --process_count $PRECESSES

python3 extract_plates.py --indir $SCALEAI --outdir $EXTRACTED --logdir $LOG_DIR --clear_outdir 1 --min_score $MIN_SCORE --process_count $PRECESSES

python3 split_train_test.py -d $EXTRACTED --trainroot $TRAIN --valroot $VALID --prev_valid $PREV_VALID --prev_train $PREV_TRAIN --split_rate $SPLIT_RATE --logdir $LOG_DIR --seed $SEED --process_count $PRECESSES

python3 mk_lmdb_dataset.py -d $VALID --lmdbdir $VALID_LMDB --min_score $MIN_SCORE

python3 augmentation.py -d $TRAIN -d $EXTRACTED_US5 -a $AUG_PER_PLATE_SCALEAI -a $AUG_PER_PLATE_US5 --stacked_coeff $STACKED_COEFF --outdir $TRAIN_AUGM --out_lmdb $TRAIN_LMDB --logdir $LOG_DIR --clear_outdir 1 --min_score $MIN_SCORE --process_count $PRECESSES


# Data: scaleAI_all_06182021
# 	Imgs: 389744
# 	Anns: 1558952

# Data: scaleAI_all_07072021
# 	Imgs: 423268
# 	Anns: 1693057

# US5
# Number of frames: 74,220
# Number of extracted plates: 117,209
# Unique plates: 31,812
# Stacked: 6036 (5.150%)

# scaleAI
# Number of frames: 423,268
# Number of extracted plates: 285,332
# Unique plates: 137,381
# Stacked: 8,301 (2.909%)

# Test data
# Count files scaleAI: 28,582

# Train data
# Count files scaleAI: 256,750 (90%)
# Count files US5: 117,209 (100%)

# Augm data
# Count generated: 4,254,181


#----------------------------------
# v7_2
# ---------------------------------
# US5
# Number of frames: 74,220
# Number of extracted plates US5: 86,432

# scaleAI
# Number of frames: 407,763
# Number of extracted plates: 240,043
