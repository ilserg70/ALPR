#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import logging

import azureml_lib as aml

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str,   default='ocr-sor-preprocess', help='Experiment name to display in Experiments tab.')
    parser.add_argument('--cluster_name',    type=str,   default='sergey-cpu', help='Name of cluster. Please refer to AzureML dashboard')
    parser.add_argument('--datastore_name',  type=str,   default='',    help='Name of datastore. Please refer to AzureML dashboard')
    parser.add_argument('--conda_cfg',       type=str,   default='config/preprocess.yml', help='Conda dependencies .yml file path.')
    parser.add_argument('--PLv',             type=str,   required=True, help='PLv version')
    parser.add_argument('--data_scaleai',    type=str,   required=True, help='Format `name:version`')
    parser.add_argument('--data_us5',        type=str,   required=True, help='Format `name:version`')
    parser.add_argument('--prev_valid',      type=str,   default='',    help="Previous validation dataset (.jpg or .mdb)")
    parser.add_argument('--prev_train',      type=str,   default='',    help="Previous training dataset (.jpg or .mdb)")
    parser.add_argument('--as_mount',        type=int,   default=1,     help='Input datasets as_mount or as_download')
    parser.add_argument('--min_score',       type=float, default=0.1,   help='Min score for plate images')
    parser.add_argument('--split_rate',      type=float, default=0.9,   help='Train/test split rate')
    parser.add_argument('--seed',            type=int,   default=100,   help='Random seed')
    parser.add_argument('--process_count',   type=int,   default=16,    help='Process count.')
    args = parser.parse_args()

    # Output registered datasets:
    TEST_DATA  = (f'ALPR_OCR_PLv{args.PLv}_test',       'ScaleAI plate images (.jpg).')
    TEST_LMDB  = (f'ALPR_OCR_PLv{args.PLv}_test_lmdb',  'ScaleAI LMDB dataset.')
    TRAIN_DATA = (f'ALPR_OCR_PLv{args.PLv}_train',      'ScaleAI plate images (.jpg).')
    TRAIN_LMDB = (f'ALPR_OCR_PLv{args.PLv}_train_lmdb', 'ScaleAI + US5 LMDB dataset.')
    US5_DATA   = (f'ALPR_OCR_PLv{args.PLv}_us5',        'US5 plate images (.jpg).')

    # Checking arguments    
    aml.check_data_version(args.data_scaleai)
    aml.check_data_version(args.data_us5)
    if args.prev_valid:
        aml.check_data_version(args.prev_valid)
    if args.prev_train:
        aml.check_data_version(args.prev_train)

    alpr = aml.ALPRPipeline(
        args.cluster_name, args.conda_cfg, 
        source_dir     = '../preprocess', 
        datastore_name = args.datastore_name, 
        allow_reuse    = False
    )

    # Input datasets
    input_scaleai = alpr.source_dataset(args.data_scaleai, args.as_mount)
    input_us5     = alpr.source_dataset(args.data_us5, args.as_mount)
    prev_valid, prev_train = None, None
    if args.prev_valid:
        prev_valid = alpr.source_dataset(args.prev_valid, args.as_mount)
    if args.prev_train:
        prev_train = alpr.source_dataset(args.prev_train, args.as_mount)

    # Pipeline data
    extract_scaleai = alpr.pipeline_dataset('extract_scaleai')
    testing_data    = alpr.pipeline_dataset('testing_data')
    testing_lmdb    = alpr.pipeline_dataset('testing_lmdb')
    training_data   = alpr.pipeline_dataset('training_data')
    training_lmdb   = alpr.pipeline_dataset('training_lmdb')
    extract_us5     = alpr.pipeline_dataset('extract_us5')

    step_extract_scaleai = alpr.step_extract_plates(input_scaleai, args.data_scaleai, extract_scaleai, args.min_score, args.process_count)
    
    step_split_train_test_scaleai = alpr.step_split_train_test(
        [extract_scaleai], training_data, testing_data, prev_valid, prev_train, 
        args.split_rate, args.seed, args.process_count)
    
    step_mk_test_lmdb = alpr.step_mk_lmdb([testing_data], testing_lmdb, args.min_score)

    step_extract_us5 = alpr.step_extract_plates(input_us5, args.data_us5, extract_us5, args.min_score, args.process_count)

    step_mk_train_lmdb = alpr.step_mk_lmdb([training_data, extract_us5], training_lmdb, args.min_score)

    step_register_dataset = alpr.step_register_dataset(
        datasets = [testing_data, testing_lmdb, training_data, training_lmdb, extract_us5], 
        names    = [TEST_DATA[0], TEST_LMDB[0], TRAIN_DATA[0], TRAIN_LMDB[0], US5_DATA[0]], 
        dscrs    = [TEST_DATA[1], TEST_LMDB[1], TRAIN_DATA[1], TRAIN_LMDB[1], US5_DATA[1]]
    )

    alpr.run_pipeline([
        step_extract_scaleai,
        step_split_train_test_scaleai,
        step_mk_test_lmdb,
        step_extract_us5,
        step_mk_train_lmdb,
        step_register_dataset

    ], args.experiment_name)
