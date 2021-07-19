# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from datetime import datetime
from time import time

# Calculation of US States similarity based on plate texts

import utils
import alphabet
import image_transform as imgtf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', action='append', required=True, help="Datasets: 1) -d PATH1 -d PATH2 .. or 2) -d NAME1:VERSION1 -d NAME2:VERSION2 ...")
    parser.add_argument('--outdir',        type=str,        required=True, help="Output directory.")
    parser.add_argument('--clear_outdir',  type=int,        default=0,     help='Clear output directory')
    parser.add_argument('--min_rate',      type=float,      default=0.5,   help="Min similarity rate (from 0 to 1)")
    parser.add_argument('--min_size',      type=int,        default=0,     help="Min namber of samples per state")
    args = parser.parse_args()

    print(f"\nCalc States similarity. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)

    outdir = utils.mk_or_clear_dir(args.outdir, bool(args.clear_outdir))

    files = utils.get_files(args.dataset, ext='.jpg')
    plates = utils.group_by_plate_text(files, show=True)

    print("\nCollecting statistics and building vectors of similarity for each State...")
    stat = {}
    for text, p in plates:
        state = p['state'] = utils.most_likely_state([i['state'] for i in p['info']])
        if state:
            if state not in stat:
                stat[state] = {'cnt': 0, 'vect': {}}
            stat[state]['cnt'] += 1
            alphabet.add_to_similarity_map(text, stat[state]['vect'], max_len=12)
    for v in stat.values():
        v['vect'] = alphabet.to_similarity_vect(v['vect'])
        
    print("\nCalculation of cosin similarity...")
    states = list(stat.keys())
    sim_map = {s1: {s2: 0 for s2 in states if s1!=s2} for s1 in states}
    sim_tbl = []
    for i in range(len(states)-1):
        s1 = states[i]
        for j in range(i+1, len(states)):
            s2 = states[j]
            rate = alphabet.cosin_similarity(stat[s1]['vect'], stat[s2]['vect'])
            sim_map[s1][s2] = sim_map[s2][s1] = rate
            sim_tbl.append((s1, s2, rate))
    
    print("\nSimilarities per State:\n")
    rows = [['State','Count']+['State','Rate']*(len(states)-1)]
    sep='\t'
    for s, v in sorted(stat.items(), key=lambda x: -x[1]['cnt']):
        row = [s, str(v['cnt'])]
        rr = sorted(sim_map[s].items(), key=lambda x: -x[1])
        for s_, rate in rr:
            row.append(s_)
            row.append(f"{rate:.3f}")
        rows.append(row)
        if v['cnt']>=args.min_size:
            rr = [(s_, rate) for s_, rate in rr if rate>=args.min_rate]
            if rr:
                top = sep.join([f"{s_:s}{sep}{rate:.3f}" for s_, rate in rr])
                print(f"{str(s):15s}{sep}{v['cnt']:d}{sep}{top}")
    utils.write_csv(outdir.joinpath("similarity_per_state.csv"), rows)

    print("\nSorted similarities:\n")
    rows = [['State','State','Rate']]
    sim_tbl = sorted(sim_tbl, key=lambda x: -x[2])
    for s1, s2, rate in sim_tbl:
        rows.append([s1, s2, f"{rate:.3f}"])
        if rate >= args.min_rate and stat[s1]['cnt']>=args.min_size and stat[s2]['cnt']>=args.min_size:
            print(f"{s1:15s}{sep}{s2:15s}{sep}{rate:.3f}")
    utils.write_csv(outdir.joinpath("similarity_sorted.csv"), rows)

    print("\nCreate examples of typical plate images per US State:\n")
    m = 3
    state_map = utils.group_by_region(plates, unique_texts=True)
    for state, files in state_map.items():
        out_file = f"{state}.jpg"
        imgtf.concat_images(files, m, outdir.joinpath(out_file))
        print(f"\t{out_file} saved")

    print(f"\nElapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
