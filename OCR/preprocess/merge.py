# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
from datetime import datetime
from time import time

from tqdm import tqdm
from joblib import Parallel, delayed

import  utils

# python3 merge.py --indir1 '/Users/silinskiy/mydata/merging/1' --indir2 '/Users/silinskiy/mydata/merging/2' --outdir '/Users/silinskiy/mydata/merging/out'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',  action='append', required=True, help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('--outdir', type=str, required=True, help="Output dir with files")
    parser.add_argument('--process_count', type=int, default=8, help="Number of processes")
    args = parser.parse_args()

    print(f"Start merging. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)

    indirs = list(map(Path, args.dataset))

    outdir = utils.mk_dir(args.outdir)

    print("Reading from input dirs...")
    copy_map = [(f, outdir.joinpath(f.name)) for d in indirs for f in d.rglob(f'*.jpg')]
    print(f"Total files: {len(copy_map)}")

    print("Run merging...")
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(shutil.copyfile)(in_path, out_path)
        for in_path, out_path in tqdm(copy_map, total=len(copy_map), mininterval=10)
    )

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
