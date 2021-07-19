# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from time import time
from pathlib import Path

import utils
import image_transform as imgtf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',  action='append', required=True, help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('--lmdbdir',    type=str,   required=True, help="Output directory with .mdb data")
    parser.add_argument('--min_score',  type=float, default=0.1,   help='Files with score less than this threshold will be discarded')
    args = parser.parse_args()

    print(f"Start creating LMDB dataset. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)

    assert 0 <= args.min_score and args.min_score < 1, f"--min_score {args.min_score} must be >= 0 and < 1"
    
    lmdbdir = utils.clear_dir(args.lmdbdir)

    plate_files = utils.get_files(args.dataset, ext='.jpg')

    print("Saving data to LMDB...")
    utils.create_lmdb(plate_files, lmdbdir, args.min_score, read_image_func=imgtf.read_image, shuffle_flag=True)

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
