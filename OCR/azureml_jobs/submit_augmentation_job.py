#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import logging

import azureml_lib as aml

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str,   default='ocr-sor-augmentation', help='Experiment name to display in Experiments tab.')
    parser.add_argument('--cluster_name',    type=str,   default='sergey-cpu', help='Name of cluster. Please refer to AzureML dashboard.')
    parser.add_argument('--datastore_name',  type=str,   default='',    help='Name of datastore. Please refer to AzureML dashboard')
    parser.add_argument('--conda_cfg',       type=str,   default='config/preprocess.yml', help='Conda dependencies .yml file path.')
    parser.add_argument('--PLv',             type=str,   required=True, help='PLv version')    
    parser.add_argument('--data_scaleai',    type=str,   required=True, help='Format `name:version`. Input plate (.jpg) images.')
    parser.add_argument('--data_us5',        type=str,   required=True, help='Format `name:version`. Input plate (.jpg) images.')
    parser.add_argument('--as_mount',        type=int,   default=1,     help='Input datasets as_mount or as_download.')
    parser.add_argument('--augs_scaleai',    type=int,   required=True, help='Number of augmentations per unique plate text.')
    parser.add_argument('--augs_us5',        type=int,   required=True, help='Number of augmentations per unique plate text.')
    parser.add_argument('--stacked_coeff',   type=float, default=1.0,   help='Factor for stacked plates')
    parser.add_argument('--min_score',       type=float, default=0.1,   help='Min plate image score to save augmentation.')
    parser.add_argument('--min_br',          type=float, default=60,    help='Min plate image brightness to save augmentation.')
    parser.add_argument('--seed',            type=int,   default=42,    help='Random seed.')
    parser.add_argument('--process_count',   type=int,   default=16,    help='Process count.')
    args = parser.parse_args()

    # Output registered datasets:
    dscr_scaleai = f'augs_per_plate: {args.augs_scaleai}'
    dscr_us5 = f'augs_per_plate: {args.augs_us5}'
    AUGM_LMDB = (f'ALPR_OCR_PLv{args.PLv}_augm_lmdb', f'Scaleai ({dscr_scaleai}) + US5 ({dscr_us5}) LMDB dataset. stacked_coeff: {args.stacked_coeff}')

    # Checking arguments
    aml.check_data_version(args.data_scaleai)
    aml.check_data_version(args.data_us5)

    alpr = aml.ALPRPipeline(args.cluster_name, args.conda_cfg, '../preprocess', args.datastore_name, allow_reuse=False)

    # Input datasets
    input_scaleai = alpr.source_dataset(args.data_scaleai, args.as_mount)
    input_us5     = alpr.source_dataset(args.data_us5, args.as_mount)

    # Pipeline data
    augm_data = alpr.pipeline_dataset('augm_data')
    augm_lmdb  = alpr.pipeline_dataset('augm_lmdb')

    step_augm = alpr.step_augmentation(
        [input_scaleai, input_us5], [args.augs_scaleai, args.augs_us5],
        augm_data, augm_lmdb,
        args.stacked_coeff, args.min_score, args.min_br, args.seed, args.process_count
    )
    
    step_register_dataset = alpr.step_register_dataset(
        [augm_lmdb], [AUGM_LMDB[0]], [AUGM_LMDB[1]], 
        tags={
            'augs_per_plate_scaleai': args.augs_scaleai,
            'augs_per_plate_us5': args.augs_us5,
            'stacked_coeff': args.stacked_coeff,
            'min_score': args.min_score,
        }
    )

    alpr.run_pipeline([
        step_augm,
        step_register_dataset
    ], args.experiment_name)
