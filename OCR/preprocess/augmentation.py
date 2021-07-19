# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import random
import shutil
from datetime import datetime
from time import time
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

import utils
import image_transform as imgtf

def create_augm_map(plates, augs_per_plate, stacked_coeff=1):
    """ Creation of augmentation map
        @param: plates - Plate images grouped by plate texts
        @param: augs_per_plate - Number of augmentations per unique plate text
        @param: stacked_coeff - Factor for stacked plates
        @return: augm_map - {file: count_augm}
    """
    augm_map = []
    for _, p in plates:
        augs_per_plate_ = augs_per_plate
        if p['stacked']:
            augs_per_plate_ = int(augs_per_plate_ * stacked_coeff)
        orig_cnt = len(p['files'])
        if orig_cnt < augs_per_plate_:
            augs_per_image = max(int(round((augs_per_plate_ - orig_cnt) / orig_cnt)), 1)
            for f in p['files']:
                augm_map.append((Path(f), augs_per_image))
    return augm_map

def generate_images(img_path, weight, out_dir, aug_log, out_files, min_score, min_br):
    """ Generate augmented images.
        @param: img_path - Path to input plate image
        @param: weight - How many images to generate
        @param: out_dir - path to output dir
        @param: aug_log - Augmentation log
        @param: out_files - Output files
        @param: min_score - Min value for score
        @param: min_br - Min brightness
        @return: None 
    """
    # Exit if no augmented weight.
    if weight == 0:
        return

    res = imgtf.do_random_transformations2(img_path, weight, out_dir, min_score, min_br)
    aug_log.append([Path(img_path).name] + [res[k] for k in ['count','required','attempts','max_attempts','msg']])
    out_files += res['files']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',  action='append', required=True, help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('-a', '--augs_per_plate',  action='append', required=True, help="Number of augmentations per unique plate text.")
    parser.add_argument('--stacked_coeff',  type=float, default=1.0,       help='Factor for stacked plates.')
    parser.add_argument('--outdir',         type=str,   required=True,     help="Output directory with augmented images.")
    parser.add_argument('--out_lmdb',       type=str,   default='',        help="Output directory with LMDB data.")
    parser.add_argument('--clear_outdir',   type=int,   default=0,         help='Clear output directory')
    parser.add_argument('--logdir',         type=str,   default='outputs', help="Output directory with logging info.")
    parser.add_argument('--min_score',      type=float, default=0.1,       help='Min value for score threshold.')
    parser.add_argument('--min_br',         type=float, default=60,        help='Min value for brightness threshold.')
    parser.add_argument('--seed',           type=int,   default=100,       help="Seed for randomization.")    
    parser.add_argument('--process_count',  type = int, default=8,         help='Number of processes.')
    args = parser.parse_args()

    print(f"\nStart augmentation. Time: {str(datetime.now())}")
    start_time = time()
    now_date = str(datetime.now().date())
    utils.print_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = [Path(d) for d in args.dataset]
    augs_per_plate = [int(a) for a in args.augs_per_plate]
    stacked_coeff = float(args.stacked_coeff)

    for d in dataset:
        assert d.exists(), f"Input directory doesn't exist: {d.name}"
    assert args.outdir, f"Error! Output dir doesn't defined: {args.outdir}"
        
    out_dir = utils.mk_or_clear_dir(args.outdir, bool(args.clear_outdir))
    out_lmdb = None
    if args.out_lmdb:
        out_lmdb = utils.mk_or_clear_dir(args.out_lmdb, True)
    log_dir = utils.mk_dir(args.logdir)

    plate_files = [utils.get_files(d, ext='.jpg') for d in dataset]
    orig_files = [f for ff in  plate_files for f in ff]
    out_files = [str(out_dir.joinpath(Path(f).name)) for ff in plate_files for f in ff]

    print("Copying original files...")
    total_orig = len(out_files)
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(shutil.copyfile)(str(img_path), out_path)
        for img_path, out_path in tqdm(zip(orig_files, out_files), total=total_orig, mininterval=max(total_orig//500, 1))
    )

    augm_map = []
    for files, augs, d in zip(plate_files, augs_per_plate, dataset):
        print("\n-------------------------------------------------------")
        print(f"Dataset: {d.name}")
        print("-------------------------------------------------------\n")
        plates = utils.group_by_plate_text(files, args.min_score, show=True)
        augm_map += create_augm_map(plates, augs, stacked_coeff)
    print(f"\nAugmented files: {sum([p[1] for p in augm_map])}")

    print("Augmentation...")
    random.shuffle(augm_map)
    aug_log = []
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(generate_images)(img_path, weight, out_dir, aug_log, out_files, args.min_score, args.min_br)
        for img_path, weight in tqdm(augm_map, total=len(augm_map), mininterval=max(len(augm_map)//500, 1))
    )
    # Logging.
    count_gen = sum([r[1] for r in aug_log])
    required  = sum([r[2] for r in aug_log])
    attempts  = sum([r[3] for r in aug_log])    
    print(f"Overall:\n\tCount generated: {count_gen}\n\tRequired: {required}\n\tAttempts: {attempts}")
    aug_tbl = pd.DataFrame(aug_log, columns=['file','count','required','attempts','max_attempts','msg'])
    tbl_path = log_dir.joinpath(f'augmented_log_{now_date}.csv')
    aug_tbl.to_csv(str(tbl_path), index=False)
    print(f"Table: aug log is saved to {tbl_path.name}")

    print("Calc. chars distributions and plotting graphs...")
    distr_orig = utils.calc_chars_distribution(utils.group_by_plate_text(orig_files, show=True))
    distr_augm = utils.calc_chars_distribution(utils.group_by_plate_text(out_files, show=True))

    utils.plot_chars_distribution(
        labels=['orig_data', 'augm_data'],
        distrs= [distr_orig, distr_augm],
        plot_file=log_dir.joinpath(f'chars_distribution_{now_date}.png')
    )

    if out_lmdb:
        print("Saving data to LMDB...")
        utils.create_lmdb(out_files, out_lmdb, args.min_score, read_image_func=imgtf.read_image, shuffle_flag=True)

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
