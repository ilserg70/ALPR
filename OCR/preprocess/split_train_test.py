# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import random
import shutil
from datetime import datetime
from time import time

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import  utils

def mk_copy_map(files, outdir):
    return [(str(f), str(outdir.joinpath(Path(f).name))) for f in files if Path(f).exists()]

def filter_off_prev_data(plates, prev_valid, prev_train, valroot, trainroot):
    """ Filter off prev. valid/train data.
        @param: plates - Plate images grouped by plate texts
        @param: prev_valid - Previous validation data
        @param: prev_train - Previous training data
        @param: valroot - Path to validation dataset.
        @param: trainroot - Path to training dataset.
        @return: copy_map, plates_rest - Copy file map and the rest of plates
    """
    copy_map = []
    plates_rest = []
    stat = {'prev_valid': 0, 'prev_train': 0, 'prev_intersect': 0}
    for text, p in plates:
        flag_valid = text in prev_valid
        flag_train = text in prev_train
        if not flag_valid and not flag_train:
            plates_rest.append((text, p))
        elif flag_valid:
            copy_map += mk_copy_map(p['files'], valroot)
            stat['prev_valid'] += len(p['files'])
        elif flag_train:
            copy_map += mk_copy_map(p['files'], trainroot)
            stat['prev_train'] += len(p['files'])
        else:
            stat['prev_intersect'] += len(p['files'])

    return copy_map, plates_rest, stat

def copyfile(in_path, out_path, err_log):
    try:
        shutil.copyfile(in_path, out_path)
    except Exception as e:
        p1, p2 = Path(in_path), Path(out_path)
        err_log.append([p1.parent.name+'/'+p1.name, p2.parent.name+'/'+p2.name, str(e)])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='append', required=True, help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('--prev_valid',    type=str,   default='',        help="Previous validation dataset (.jpg or .mdb)")
    parser.add_argument('--prev_train',    type=str,   default='',        help="Previous training dataset (.jpg or .mdb)")
    parser.add_argument('--trainroot',     type=str,   required=True,     help="Output directory with .jpg data for training")
    parser.add_argument('--valroot',       type=str,   required=True,     help="Output directory with .jpg data for validation")
    parser.add_argument('--logdir',        type=str,   default='outputs', help="Output directory with logging info." )
    parser.add_argument('--seed',          type=int,   default=42,       help="Seed for randomization")
    parser.add_argument('--split_rate',    type=float, default=0.9,       help="Example: 0.9 - 90%% for training, 10%% for testing.")
    parser.add_argument('--process_count', type=int,   default=8,         help="Number of processes")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Splitting data to training and testing datasets. Time: {str(datetime.now())}")
    start_time = time()
    now_date = str(datetime.now().date())
    utils.print_args(args)

    # Check arguments
    split_rate = float(args.split_rate)
    assert split_rate > 0.0 and split_rate < 1.0, f"Error! --split_rate {args.split_rate} must be > 0 and < 1"
    for d in args.dataset + [args.prev_valid, args.prev_train]:
        assert not d or Path(d).exists(), f"Error! input dir doesn't exist: {d}"
    
    trainroot = utils.clear_dir(args.trainroot)
    valroot = utils.clear_dir(args.valroot)
    log_dir = utils.mk_dir(args.logdir)

    # Get input files
    plate_files = utils.get_files(args.dataset, ext='.jpg')

    # Get plates sorted by count of files desc.
    plates = utils.group_by_plate_text(plate_files, show=True)
    count = sum([len(p['files']) for _, p in plates])
    
    print(f"split_rate: {split_rate}")
    copy_map = []

    # Get files to exclude
    if args.prev_valid or args.prev_train:
        prev_valid = dict(utils.group_by_plate_text(args.prev_valid)) if args.prev_valid else {}
        prev_train = dict(utils.group_by_plate_text(args.prev_train)) if args.prev_train else {}

        # Filter off exclude plates
        copy_map, plates, stat = filter_off_prev_data(plates, prev_valid, prev_train, valroot, trainroot)
        if len(copy_map)>0:
            split_rate = (split_rate * count - stat['prev_train']) / (count - len(copy_map))
            print(f"Using:\n\tprev_valid: {Path(args.prev_valid).name}\n\tprev_train: {Path(args.prev_train).name}")
            print(f"\t{stat}")
            print(f"New split_rate: {split_rate:0.3f}")

    # Splitting preparation fore the rest files
    for text, p in plates:
        outdir = trainroot if random.uniform(0, 1) < split_rate else valroot
        copy_map += mk_copy_map(p['files'], outdir)

    print("Run splitting...")
    err_log = [['Source','Destination','Error']]
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(copyfile)(in_path, out_path, err_log)
        for in_path, out_path in tqdm(copy_map, total=len(copy_map), mininterval=max(len(copy_map)//500, 1))
    )
    if len(err_log)>1:
        log_file = log_dir.joinpath(f"errors_log.csv")
        utils.write_csv(log_file, err_log)
        print(f"Errors of copying: {len(err_log)-1} of total copying: {len(copy_map)}")
        assert len(err_log)-1 >= 0.5 * len(copy_map), "Errors of copying is greater than 50%"
    
    # Check results of splitting
    train_files = []
    test_files = []
    for _, out_path in copy_map:
        p = Path(out_path)
        if p.parent.name == trainroot.name:
            train_files.append([True, p.name])
        else:
            test_files.append([False, p.name])

    n_train, n_test = len(train_files), len(test_files)
    print(f"Resulted split_rate: {n_train/(n_train  + n_test):0.3f}")

    log_file = f"split_log_{now_date}.csv"
    print(f"Save split log: {log_dir.name}/{log_file}")
    tbl = pd.DataFrame(
        train_files + test_files, 
        columns = ['Train', 'File']
    )
    tbl.to_csv(str(log_dir.joinpath(log_file)), index=False)

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
