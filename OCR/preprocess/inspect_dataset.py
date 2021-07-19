# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from datetime import datetime
from time import time

import utils

# Inspect datasets and calc. intersection. Datasets: .jpg or .mdb or mix

# Run:
# python3 preprocess/inspect.py -d DIR1 -d DIR2 ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='append', required=True, help="Datasets: -d PATH1 -d PATH2 ..")
    parser.add_argument('--logdir', type=str, default='outputs', help='Logging dir.')
    args = parser.parse_args()

    print(f"Start inspect datasets. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)
    now_date = str(datetime.now().date())

    log_dir = utils.mk_dir(args.logdir)

    names = [Path(d).name for d in args.dataset]
    res = utils.intersect(args.dataset, names, show=True)

    for stat_name, tbl in res.items():
        print(f"\n{stat_name}:")
        print(tbl)
        tbl_path = log_dir.joinpath(f"{stat_name}_{now_date}.csv")
        tbl.to_csv(str(tbl_path), index=False)

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")