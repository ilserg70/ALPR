# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import json
import random as rm
import csv
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
import pandas as pd
import lmdb
from tqdm import tqdm

import alphabet

# Components of plate file name: "LABEL_STATE_BLOCK_SCORE_HASH_IDX.jpg"
#               Plate text,    US State name, Block #,        Image score,      Frame ID,      Index of augm.
PLATE_INFO = [('label', str), ('state', str), ('block', int), ('score', float), ('hash', str), ('idx', str)]

COLORS = [
    ["red","brown","orange","indianred","lightcoral","rosybrown","chocolate","sandybrown","peru","orangered"],
    ["green","lime","aquamarine","darkgreen","forestgreen","limegreen","mediumseagreen","turquoise","olive","yellowgreen"],
    ["blue","aqua","darkturquoise","cadetblue","powderblue","deepskyblue","skyblue","mediumblue","steelblue","dodgerblue"],
    ["orange","yellow","bisque","khaki","burlywood","tan","navajowhite","darkgoldenrod","goldenrod","gold"],
    ["magenta","hotpink","crimson","lightpink","deeppink","mediumvioletred","orchid","purple","plum","violet"]
]

def get_color(i):
    COLORS[i%len(COLORS)][(i//len(COLORS))%len(COLORS[0])]

def print_args(args):
    """ Print args of Argument Parser
    """
    print("Input arguments:")
    for arg in vars(args):
        print("  --{0:20s}  {1:}".format(arg, getattr(args, arg)))

def mk_or_clear_dir(dir_path, clear_flag=False):
    return clear_dir(dir_path) if clear_flag else mk_dir(dir_path)

def mk_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def clear_dir(dir_path):
    """ Create or clear dir.
        If not exists - create, otherwise - remove all content: files and subdirs.
        @return: Path(dir_path)
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path = mk_dir(dir_path)
    elif dir_path.is_dir():
        print(f"Cleaning of: {dir_path}")
        files = list(dir_path.glob("*"))
        for f in tqdm(files, total=len(files), mininterval=max(len(files)//500, 1)):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(str(f))
    return dir_path

def save_to_json(data, file_path):
    with open(str(file_path), 'w') as fp:
        json.dump(data, fp)

def read_from_json(file_path):
    with open(str(file_path), 'r') as fp:
        return json.load(fp)

def save_txt(file_path, data_list):
    with open(str(file_path), 'w') as fp:
        fp.writelines([f"{d}\n" for d in data_list])

def read_txt(file_path):
    with open(str(file_path), 'r') as fp:
        return [row.strip() for row in fp.readlines()]

def write_csv(csv_file, rows):
    """ Writing data to .csv file.
    """
    with open(str(csv_file), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)

def get_files(source, ext='.jpg'):
    """ Get all files from in_dir and sub dirs.
            @param: source - Input dir or list of dirs
            @param: ext - extention, for example: '.txt'
            @return: files - List of files
    """
    files = []
    for d in source if isinstance(source, list) else [source]:
        if d:
            print(f"Getting {ext} files from: {Path(d).name}")
            p = Path(d)
            assert p.exists(), f"Error! Dir doesn't exist: {d}"
            files_ = [str(f) for f in p.rglob(f'*{ext}')]
            print(f"Count files: {len(files_)}")
            files += files_
    files = list(set(files))
    print(f"Total files: {len(files)}")
    return files

def mk_file_name(info):
    """ Make file name for plate image. It includes all annotation info.
    @param: info - dict with keys: PLATE_INFO
    @return: file_name
    """
    xx = [t(info[name]) for name, t in PLATE_INFO]
    xx = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in xx]
    return '_'.join(xx) + '.jpg'

def split_name(file_path):
    """ Extract annotation info from plate image file name.
        Example: Y9728_California_3_0.148_98627502-20d8-4cc0-8c71-c0544471a1a8_0.jpg
        @param: file_path - Path to plate image
        @return: info - dict
    """
    xx = Path(file_path).stem.split('_')
    info = {name: t(x) for (name, t), x in zip(PLATE_INFO, xx)}
    return info

def correct_info(info):
    for name, t in PLATE_INFO:
        val = info.get(name, None)
        info[name] = t(val) if val is not None or t==str else 1.0 if name=='score' else 0
    return info

def get_not_existed_path(out_dir, info):
    plate_path = out_dir.joinpath(mk_file_name(info))
    while plate_path.exists():
        info['idx'] = str(info['idx']) + '0'
        plate_path = out_dir.joinpath(mk_file_name(info))
    return plate_path

def find_value(obj, name):
    """ Find first value by name
    @param: obj - Object type of: list, dict
    @param: name - Key
    @return value
    """
    try:
        if isinstance(obj, dict):
            for k, o in obj.items():
                if k==name:
                    return o
                v = find_value(o, name)
                if v:
                    return v
        elif isinstance(obj, list):
            for o in obj:
                v = find_value(o, name)
                if v:
                    return v
    except Exception:
        pass
    return None

def create_lmdb(files, outdir, min_score, read_image_func, shuffle_flag=True):
    """Create LMDB dictionary type.
        @param: files - List of paths to plate image files
        @param: outdir - Path to output dir
        @param: min_score - Score threshold (min value)
        @param: read_image_func - Function to get image
        @return: None
    """
    if shuffle_flag:
        rm.shuffle(files)
    num = len(files)
    db = lmdb.open(str(outdir), map_size=1099511627776)
    cache = {}
    max_cache_size = 1000
    count = 0
    for img_file in tqdm(files, total=len(files), mininterval=max(len(files)//500, 1)):
        info = split_name(img_file)
        if info['score'] >= min_score:
            image_bin = read_image_func(img_file)
            if image_bin is not None:
                count += 1 # 1-based numeration
                key = "{:09d}".format(count)
                cache[f"image-{key}"] = image_bin
                for name_, _ in PLATE_INFO:
                    cache[f"{name_}-{key}"] = str(info[name_])
                if len(cache) >= max_cache_size:
                    write_cache(db, cache)
                    cache = {}
    cache['num-samples'] = str(count)
    write_cache(db, cache)
    cache = {}
    db.close()
    print(f"Created LMDB - Original files: {num}, Saved to LMDB: {count}, Corrupted images: {num - count}")

def write_cache(env, cache):
    """ Write cache into LMDB object
            env - import lmdb; env = lmdb.open(...)
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(k, str):
                k = k.encode()
            if isinstance(v, str):
                v = v.encode()
            txn.put(k,v)

def get_num_samples(source):
    """ Get number of images
            @param: source - Dir with image files (.jpg) or LMDB files (.mdb)
            @return: num_samples
    """
    files = get_files(source, ext='.jpg')
    num_samples = len(files)
    if num_samples==0 and len(get_files(source, ext='.mdb')) > 0:
        data = lmdb.open(str(source), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with data.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode('utf-8')))
    return num_samples

def group_by_plate_text(source, min_score=0, show=False):
    """ Group licence plate files by licence plate text.
            @param: source - 1) List of image paths or 2) dir with image files or 3) LMDB dir
            @param: min_score - Threshold for min score
            @param: show - To print or not to print results
            @return: plates - Sorted items by count of plate files desc.
    """
    if isinstance(source, list) and len(source)==0:
        return []

    files = source if isinstance(source, list) else get_files(source, ext='.jpg')

    plates = {}

    def _add(info, file_name, plates, min_score):
        if info['score'] >= min_score:
            text = info['label']
            if text:
                if text not in plates:
                    plates[text] = {'files': [], 'info': [], 'stacked': alphabet.is_stacked(text)}
                plates[text]['files'].append(file_name)
                plates[text]['info'].append(info)

    # Image files from list source or from input dir 
    if files:
        for file_name in files:
            info = split_name(file_name)
            _add(info, file_name, plates, min_score)
    # Plate samples from LMDB
    elif len(get_files(source, ext='.mdb')) > 0:
        data = lmdb.open(str(source), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        def _get_value(txn, name, key, default=None):
            value  = txn.get(f"{name}-{key}".encode('utf-8'))
            return value.decode("utf-8") if value is not None else default

        with data.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode('utf-8')))
            for index in range(num_samples):
                key = "{:09d}".format(index+1) # 1-based numeration
                info = correct_info({name: _get_value(txn, name, key, None) for name, _ in PLATE_INFO})
                file_name = mk_file_name(info)
                _add(info, file_name, plates, min_score)

    plates = sorted(plates.items(), key=lambda item: -len(item[1]["files"]))
    if show:
        print_plates_statistics(plates)
    return plates

def group_by_region(plates, select_states=None, unique_texts=False):
    """ Group plate files by US State and sort by score desc.
    @param: plates - Plates grouped by plate text
    @param: select_states - Filter for pre-selected States
    @param: unique_texts - Select only unique plate texts
    @return: state_map - Plate files grouped by US State
    """
    state_map = {}
    if isinstance(plates, dict):
        plates = plates.items()
    for _, p in plates:
        fi = sorted(zip(p['files'], p['info']), key=lambda p: -p[1]['score'])
        if unique_texts:
            fi = fi[:1]
        for f, i in fi:
            s = i['state']
            if select_states and s not in select_states:
                continue
            if s not in state_map:
                state_map[s] = []
            state_map[s].append((f, i['score']))
    for s in state_map.keys():
        state_map[s] = [f for f, _ in sorted(state_map[s], key=lambda p: -p[1])]
    return state_map

def intersect(source, name, show=False):
    """ Calc intersection of two datasets.
        @param: source - List of: 1) dirs 2) list of files
        @param: name - List of names
        @return: total, inter, pct
    """
    m = len(source)
    data = {}
    for d, n in zip(source, name):
        if isinstance(d, dict):
            data[n] = d
        else:
            if show:
                print("\n-------------------------------------------------------")
                print(f"Dataset: {n}")
                print("-------------------------------------------------------")
            data[n] = dict(group_by_plate_text(d, show=show))

    print("Calc. intersection ...")
    inter = {i: {j: [0, 0] for j in range(m)} for i in range(m)} # [all, unique]
    for i in range(m):
        for j in range(i, m):
            for text, p1 in data[name[i]].items():
                if i==j:
                    inter[i][j][0] += len(p1['files'])
                    inter[i][j][1] += 1
                elif text in data[name[j]]:
                    p2 = data[name[j]][text]
                    inter[i][j][0] += len(p1['files'])
                    inter[i][j][1] += 1
                    inter[j][i][0] += len(p2['files'])
                    inter[j][i][1] += 1
    
    print("Calc. sub-intersection ...")
    plate = {}
    for i in range(m):
        for text, p in data[name[i]].items():
            if text not in plate:
                plate[text] = []
            plate[text].append(name[i])
    sub_inter = {}
    for text, nn in plate.items():
        key = ','.join(nn)
        if key not in sub_inter:
            sub_inter[key] = {n: [0, 0] for n in nn} # [all, unique]
        for n in nn:
            sub_inter[key][n][0] += len(data[n][text]['files'])
            sub_inter[key][n][1] += 1
    
    res = {
        'inter_all': pd.DataFrame(
            [[name[i]]+[inter[i][j][0] for j in range(m)] for i in range(m)], 
            columns = ['Dataset'] + name
        ),
        'inter_all_rate': pd.DataFrame(
            [[name[i]]+[inter[i][j][0]/max(inter[i][i][0], 0) for j in range(m)] for i in range(m)], 
            columns = ['Dataset'] + name
        ),
        'inter_uniq': pd.DataFrame(
            [[name[i]]+[inter[i][j][1] for j in range(m)] for i in range(m)], 
            columns = ['Dataset'] + name
        ),
        'inter_uniq_rate': pd.DataFrame(
            [[name[i]]+[inter[i][j][1]/max(inter[i][i][1], 0) for j in range(m)] for i in range(m)], 
            columns = ['Dataset'] + name
        ),
        'sub_inter': pd.DataFrame(
            [[key] + [ii.get(n, [0, 0])[0] for n in name]  for key, ii in sub_inter.items()], 
            columns = ['Intersect'] + name
        ),
        'sub_inter_uniq': pd.DataFrame(
            [[key] + [ii.get(n, [0, 0])[1] for n in name]  for key, ii in sub_inter.items()], 
            columns = ['Intersect'] + name
        ),
    }

    return res

def print_plates_statistics(plates):
    if isinstance(plates, dict):
        plates = sorted(plates.items(), key=lambda item: -len(item[1]["files"]))

    print("Top plate duplicates:")
    count=10
    print(f"\t{'Text':15s}\t{'Duplicates':10s}")
    for text, p in plates[:count]:
        print(f"\t{text:15s}\t{len(p['files']):10d}")
    if len(plates)>count:
        print("...")
    print(f"Unique plates: {len(plates)}")
    count_total =sum([len(p['files']) for _, p in plates])
    count_stacked = sum([len(p['files']) for _, p in plates if p['stacked']])
    print(f"Stacked: {count_stacked} ({100*count_stacked/count_total:.3f}%)")
    print(f"Total  images: {count_total}")

    print('\nStatistics per count of text duplicates:')
    stat = dict(Counter([len(p['files']) for _, p in plates]))
    print(f"{'Duplicates':10s}\t{'Count':10s}\t{'Comul.%':10s}")
    comul = 0
    total = sum(stat.values())
    for cnt, m in sorted(stat.items(), key=lambda x: -x[1]):
        comul += m
        pc = 100*comul/total
        if pc>90:
            break
        print(f"{cnt:10d}\t{m:10d}\t{100*comul/total:5.1f}%")
    print("...")

    for name in ['state','block']:
        print(f"\n{name.title()}s distribution:")
        print(f"{name.title():20s}\t{'Count':6s}")
        for e in Counter([i[name] for _, p in plates for i in p['info']]).items():
            print(f"{str(e[0]):20s}\t{e[1]:6d}")
    
    print("\nScores histogram:")
    bins = 10
    print(f"{'Range':14s}\t{'Count':6s}")
    h = np.histogram([i['score'] for _, p in plates for i in p['info']], bins=bins)
    for i in range(bins):
        print(f"[{h[1][i]:5.3f}, {h[1][i+1]:5.3f}]\t{h[0][i]:6d}")

def chunks(lst, n):
    """ Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split_data_to_groups(data, num_groups):
    """ Split data to chunks.
        @param: data - data list
        @param: num_groups - number of groups
    """
    return list(chunks(data, n=len(data)//num_groups+1))

def split_data_to_chunks(data, folds, seed=None):
    """ Split data to chunks.
        @param: data - data
        @param: folds - number of folds
    """
    # Split data to chunks of size = folds
    data_chunks = list(chunks(data, folds))

    if seed is not None and isinstance(seed, int):
        rm.seed(seed)

    # Shuffle data inside each chunk
    for chunk in data_chunks:
        rm.shuffle(chunk)

    return data_chunks

def print_data_chunks(plate_chunks, count=10):
    print("Split data to chunks:")
    for i, chunk in enumerate(plate_chunks[:count]):
        print(f"  chunk {i+1}")
        print("\tfold\ttext\tduplicates")
        for fold, (text, info) in enumerate(chunk):
            print(f"\t{fold}\t{text}\t{len(info['files'])}")

def levenshtein_ratio_and_distance(s, t, ratio_calc=True):
    """ Calculates levenshtein distance or ratio between two strings.
            If ratio_calc = True, the function computes the
            levenshtein distance ratio of similarity between two strings
            For all i and j, distance[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
    """
    rows, cols = len(s)+1, len(t)+1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions, insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                cost = 2 if ratio_calc else 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    dist = distance[rows - 1][cols - 1]
    return ((len(s)+len(t)) - dist) / (len(s)+len(t)) if ratio_calc else dist

def format_time(t):
    ss = int(round(t))
    mm = ss//60
    ss -= mm*60
    hh = mm//60
    mm -= hh*60
    dd = hh//24
    hh -= dd*24
    if dd>0:
        return f"{dd:d} days, {hh:d} hours"
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def to_hist(elements, hist):
    for e in elements:
        if e not in hist:
            hist[e] = 0
        hist[e] += 1
    return hist

def norm_hist(hist):
    num = max(sum(hist.values()), 1)
    return {k: v/num for k, v in hist.items()}

def most_likely_state(states):
    empty = set([str(None), None, ''])
    h = dict(Counter([s for s in states if s not in empty]))
    if len(h)==1:
        return list(h.keys())[0]
    if len(h)==0 or len(h)>=3:
        return ''
    hh = sorted(h.items(), key=lambda x: -x[1])
    if len(hh)==2 and hh[1][1]/hh[0][1] > 1/4:
        return ''            
    return hh[0][0]

def calc_chars_distribution(plates):
    """ Calc chars (of plate texts) distribution.
        @param: plates - Image files, grouped by plate text
        @return: distr - Normalized chars distribution.
    """
    distr = {}
    if isinstance(plates, dict):
        plates = plates.items()
    for text, p in plates:
        p['chars'] = alphabet.get_chars(text)
        for c in p['chars']:
            if c not in distr:
                distr[c] = 0
            distr[c] += len(p['files'])
    count = sum(distr.values())
    # Normalization
    distr = {c: m/count for c, m in distr.items()}
    return distr

def print_chars_distribution(distr):
    row = []
    for c, prob in sorted(distr.items(), key=lambda item: -item[1]):
        if len(row)>=10:
            print(f"\t{'  '.join(row)}")
            row = []
        else:
            row.append(f"{c:2s}: {prob:0.4f}") 
    if row:
        print(f"\t{'  '.join(row)}")

def plot_chars_distribution(labels, distrs, plot_file):
    """ Plotting many chars distributions on one graph.
        @param: labels - Labels for legend
        @param: distrs - List of chars distributions
        @param: plot_file - Path to output graph image.
        @return None
    """
    distr = {}
    for distr_ in distrs:
        for c, n in distr_.items():
            if c not in distr:
                distr[c] = 0
            distr[c] += n

    common_distr = sorted(distr.items(), key=lambda x: -x[1])
    chars = [x[0] for x in common_distr]

    plt.figure(figsize=(20, 5))

    for i, (lbl_, distr_) in enumerate(zip(labels, distrs)):
        probs = [distr_.get(c, 0) for c in chars]
        plt.plot(chars, probs, label=lbl_, color=get_color(i))
    
    plt.xlabel('Chars (underscore points to small chars)')
    plt.ylabel('Probability')
    plt.title('Chars distribution')
    plt.legend()
    plt.savefig(str(plot_file))
