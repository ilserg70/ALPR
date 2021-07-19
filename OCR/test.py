#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from time import time

import pandas as pd

from tool import utils
import ocr_lib

if __name__ == "__main__":
    parser = ArgumentParser(description='OCR-SOR training/validation arguments.')
    parser.add_argument('--tag',           type=str,        default='ocr_sor', help="Output files suffix.")
    parser.add_argument('--cfg',           type=str,        required=True,     help="Net module name in ./models/ or path to .py module.")
    parser.add_argument('--hvd',           type=str,        default='False',   help="Whether to execute on azureml with horovod.")
    parser.add_argument('--seed',          type=str,        default='42',      help='Random seed number')
    parser.add_argument('-m', '--model',   action='append', required=True,     help="Models: 1) -m PATH1 -m PATH2 .. or 2) -m NAME1:VERSION1 -m NAME2:VERSION2 ...")
    parser.add_argument('-d', '--dataset', action='append', required=True,     help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('--device',        type=str,        default='gpu',     help='Cuda device: CPU or GPU (max available nodes) or 0 or 1 or 0,1 or 0,1,2,3 or ... (choose nodes)')
    parser.add_argument('--num_workers',   type=str,        default='8',       help="Number of data loader workers")
    parser.add_argument('--batch_size',    type=str,        default='256',     help="Batch size")
    parser.add_argument('--focal_loss',    type=str,        default='False',   help="If using focal loss. Values: 1,T,True,true,0,F,False,false")
    parser.add_argument('--gamma',         type=str,        default= '0.5',    help="Gamma of focal loss")
    parser.add_argument('--outdir',        type=str,        default='outputs', help="Output directoty with: model weights & logs.")
    parser.add_argument('--error_folder',  type=str,        default='',        help="Path to save plate images with errors and debugging logs")
    args = parser.parse_args()

    print(f"Start model(s) validation. Time: {str(datetime.now())}")
    start_time = time()
    ocr_lib.print_args(args)
    now_date = str(datetime.now().date())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Init horovod
    ocr_lib.activate_hvd(ocr_lib.to_bool(args.hvd))
    if ocr_lib.ONLINE:
        print("Execution on AzureML")
    if ocr_lib.IS_HVD:
        print("Using distributed claud calculation - horovod")

    device = ocr_lib.select_device(args.device, batch_size=int(args.batch_size))
    ocr_lib.set_random_seed(int(args.seed), device)

    net_module, net_type = ocr_lib.import_net_module(args.cfg)
    
    params = {
        'now_date':         now_date,        
        'net_type':         net_type,
        'imgWH':            (net_module.imgW, net_module.imgH),
        'is_RGB':           (net_module.nc==3),
        'net_label':        net_module.label,
        'net_n_val_disp':   net_module.n_val_disp,
        'classify_lambda':  net_module.classify_lambda,
        'device':           device,
        'batch_size':       int(args.batch_size),
        'num_workers':      int(args.num_workers),
        'focal_loss':       ocr_lib.to_bool(args.focal_loss),
        'gamma':            float(args.gamma),
        'converter':        utils.strLabelConverter(net_module.alphabet, device), # Convert between str and label.
        'criterion':        ocr_lib.criterion_init(net_module, args.focal_loss, device),
        'criterion_1':      ocr_lib.criterion1_init(),
        'error_folder':     ocr_lib.clear_dir(args.error_folder),
        'show_details':     True
    }

    # Models validation
    model_names = [Path(m).stem for m in args.model]
    dataset_names = [Path(d).stem for d in args.dataset]
    metrics = {d: {m: None for m in model_names} for d in dataset_names}
    for weights_source, m_name in zip(args.model, model_names):
        print(f"\nModel: {m_name}")
        model = ocr_lib.init_model(net_module)
        ocr_lib.load_model_weights(model, weights_source, device)
        model = ocr_lib.model_prepare(model, net_module, device)
        for data_source, d_name in zip(args.dataset, dataset_names):
            print(f"\nDataset: {d_name}")
            data_loader = ocr_lib.test_data_loader(data_source, params)
            metrics[d_name][m_name] = ocr_lib.validation(model, data_loader, params, msg='Testing', show_details=params['show_details'])
    
    # Table: Accuracy
    tbl_acc = []
    for d in dataset_names:
        mtrs = metrics[d]
        d_size = mtrs[model_names[0]]['total']['plate_total']
        for k1 in ['total','nostack','stack']:
            for k2 in ['plate_acc','plate_acc_0=O','char_acc']:
                tbl_acc.append([d, d_size, f"{k1}_{k2}"] + [mtrs[m][k1][k2] for m in model_names])
    tbl_acc = pd.DataFrame(tbl_acc, columns=['Dataset','Size','Metric'] + model_names)
    tbl_acc_path = outdir.joinpath(f"all_acc_{args.tag}_{now_date}.csv")
    tbl_acc.to_csv(str(tbl_acc_path), index=False)
    print(f"Table: All acc. is saved to {tbl_acc_path.name}")

    # Tables: Accuracy per US State, Block, Score
    if params['show_details'] and not ocr_lib.IS_HVD:
        for k, vv in [ ('state', ocr_lib.US_STATES), ('block', ocr_lib.BLOCKS), ('score', ocr_lib.SCORES) ]:
            tbl = []
            for v in vv:
                tbl.append([v] + [metrics[d][m][k].get(v,{}).get(j,0) for m in model_names for d in dataset_names for j in ['total','acc']])
            tbl = pd.DataFrame(tbl, columns=[k.title()] + ['&'.join([m,d,j]) for m in model_names for d in dataset_names for j in ['Count','Acc']])
            tbl_path = outdir.joinpath(f"{k}_acc_{args.tag}_{now_date}.csv")
            tbl.to_csv(str(tbl_path), index=False)
            print(f"Table: {k.title()} acc. is saved to {tbl_path.name}")

    print(f"Elapsed time: {ocr_lib.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
