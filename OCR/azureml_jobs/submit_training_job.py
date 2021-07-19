#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on example: https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/machine-learning/how-to-train-pytorch.md
"""

from argparse import ArgumentParser
import logging

from constants import TRAIN_TAG, TEST_TAG, PARENT_MODEL_TAG, SCRIPT_TAG, MSG_TAG

import azureml_lib as aml

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='alpr-ocr', help='Experiment name to display in Experiments tab.')
    parser.add_argument('--cluster_name',    type=str, default='gpu-cluster', help='Name of cluster. Please refer to AzureML dashboard')
    parser.add_argument('--conda_cfg',       type=str, default='config/training.yml', help='Conda dependencies .yml file path.')
    parser.add_argument('--train_cfg',       type=str, default='config/training.ini', help='ALPR OCR Model training parameters config file path.')
    parser.add_argument('--cfg',             type=str, required=True,  help="Model type: GlamdringV3, GlamdringV10, ...")
    parser.add_argument('--model',           type=str, default='', help='Registered (pre-trained) model name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--trainroot',       type=str, required=True,  help='Training data name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--valroot',         type=str, required=False, help='Testing data name and version. Format `name:version` or `name` (default version=latest)')
    parser.add_argument('--as_mount',        type=int, default=1, help='Input datasets as_mount or as_download')
    parser.add_argument('--node_count',      type=int, default=4,  help='Number of AzureML nodes for distributed training with horovod.')
    parser.add_argument('--process_count',   type=int, default=1,  help='Process count per node for distributed training with horovod.')
    parser.add_argument('--msg',             type=str, default='', help='Experiment message.')
    parser.add_argument('--tag',             type=str, default='', help='Tag')
    args = parser.parse_args()

    aml.check_data_version(args.trainroot)
    if args.valroot:
        aml.check_data_version(args.valroot)
    
    script_params = aml.get_script_params(args.train_cfg, args.cfg)

    alpr = aml.ALPRTrain(
        args.cluster_name, 
        args.conda_cfg, 
        source_dir= '../', 
        env_name = "ALPR-Pytorch1.7-Cuda11-OpenMpi4.1.0-py36",
        is_register_env=True,
        curated_env_name="AzureML-Pytorch1.7-Cuda11-OpenMpi4.1.0-py36", 
        env_vars={
            'AZUREML_DATASET_HTTP_RETRY_COUNT': 10000,
            'NCCL_DEBUG': 'DEBUG',
            'NCCL_RINGS': ','.join(map(str, range(args.node_count))),
            'HOROVOD_WITH_PYTORCH': 1
        }, 
        pip_pkgs=[
            'pillow==8.1.0',
            'opencv-python-headless==4.5.1.48',
            'tensorboardX',
            'lmdb',
        ]
    )

    alpr.run(
        args.experiment_name, 
        model_name    = args.cfg,
        trainroot     = args.trainroot,
        valroot       = args.valroot,
        as_mount      = args.as_mount,
        parent_model  = args.model,
        script_params = script_params,
        tags = {
            TRAIN_TAG: args.trainroot,
            TEST_TAG: args.valroot,
            PARENT_MODEL_TAG: args.model,
            SCRIPT_TAG: str(__file__),
            MSG_TAG: args.msg
        }, 
        node_count    = args.node_count, 
        process_count = args.process_count,
        tag           = args.tag
    )
