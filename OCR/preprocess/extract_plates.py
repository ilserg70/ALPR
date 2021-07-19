# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from time import time
from pathlib import Path
import shutil
import json
import traceback
import re
import uuid
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from azureml.core.run import Run
from azureml.core import Dataset

import alphabet
import utils
import image_transform as imgtf
import geo_names

EXP_RUN    = Run.get_context() # Current experiment Run
print(f"Run_ID: {EXP_RUN._run_id}") # OfflineRun_d5c670ad-929b-40f0-a1ea-baac0315bcd9
ONLINE   = 'OfflineRun' not in EXP_RUN._run_id

WORKSPACE  = None
EXP_NAME   = None
RUN_NUMBER = None
EXP_TAGS   = None

if ONLINE:
    WORKSPACE  = EXP_RUN._experiment._workspace
    EXP_NAME   = EXP_RUN._experiment._name
    RUN_NUMBER = EXP_RUN._run_number
    EXP_TAGS   = EXP_RUN.get_tags()

LOG_CONFIG = [
    ('id','str',''), ('frame_id','str',''), ('ann_status','str',''),
    ('block','int32',0), ('condition','str',''), ('scenario','str',''), ('speed','str',''), ('location','str',''),
    ('text_ann','str',''), ('text','str',''), ('is_stacked','bool',False), ('text_len','int32',0),
    ('state_ann','str',''), ('state','str',''), ('state_fixed','bool',False),
    ('score','float64',0), ('blur_score','float64',0), ('area_score','float64',0), ('pixels_score','float64',0),
    ('area','float64',0), ('h','float64',0), ('w','float64',0), 
    ('x1_ann','float64',0), ('y1_ann','float64',0), ('x2_ann','float64',0), ('y2_ann','float64',0), ('x3_ann','float64',0), ('y3_ann','float64',0), ('x4_ann','float64',0), ('y4_ann','float64',0),
    ('x1','float64',0), ('y1','float64',0), ('x2','float64',0), ('y2','float64',0), ('x3','float64',0), ('y3','float64',0), ('x4','float64',0), ('y4','float64',0),
    ('dir','str',''), ('subdir','str','') , ('file','str',''), ('err_msg','str',''),
    ('frame_file','str','')
]
LOG_HEADER = [c for c, _, _ in LOG_CONFIG]
LOG_IDX = {name: i for i, name in enumerate(LOG_HEADER)}
LOG_TYPES = [t for _, t, _ in  LOG_CONFIG]
LOG_DEFAULTS = [d for _, _, d in LOG_CONFIG]

ANN_STATUS = {
    'passed': ['ConfirmedPassed', 'Fixed', 'Passed'],
    'failed': ['ConfirmedFailed', 'Failed'],
    'pending': ['Pending']
}

def save_log(log, log_file, indir):
    """ Cleaning and saving the extraction log
    """
    # Making relative paths to frame files
    indir = str(Path(indir))
    f = LOG_IDX['frame_file']
    for row in log:
        row[f] = row[f].replace(indir,'')
    
    print(f"Saving log table into: {log_file.name}")
    df = pd.DataFrame(log, columns=LOG_HEADER)
    for c, t, d in LOG_CONFIG:
        try:
            df[c] = df[c].fillna(d)
            df[c] = df[c].astype(t)
        except Exception as e:
            print(f"Error! Column: {c} - {str(e)}")
    df.to_csv(str(log_file), index=False)
    return df

def clean_str(s):
    s = re.sub(r'[\n\t\r,]+', ' ', s)
    s = re.sub(r'[ ]+', ' ', s)
    return s.strip()

def add_val(row, name, val):
    i = LOG_IDX[name]
    t = LOG_TYPES[i]
    d = LOG_DEFAULTS[i]
    try:
        if not val:
            row[i] = d
        elif t in ['int32','int',int]:
            row[i] = int(val)
        elif t in ['float64','float',float]:
            row[i] = float(val)
        elif t in ['bool',bool]:
            row[i] = bool(val)
        else:
            row[i] = clean_str(str(val))
    except Exception:
        row[i] = d
    return row[i]

def add_error_msg(row, msg):
    if msg:
        i = LOG_IDX['err_msg']
        if row[i]:
            row[i] += '|' + clean_str(msg)
        else:
            row[i] = clean_str(msg)

def group_annotations_by_frames(img_files, ann_files):
    """ Grouping annotation files by frame images.
        @param: img_files
        @param: ann_files
        @return: img_ann - List of (imag_path, [annotation paths])
    """    
    # Make annotation map
    ann_map = {}
    for f in ann_files:
        frame_id = f.stem.split('_')[0]
        if frame_id not in ann_map:
            ann_map[frame_id] = []
        ann_map[frame_id].append(f)
    # Group annotation files by frame images
    img_ann = []
    for p in img_files:
        ann = ann_map.get(p.stem, []) # suffixes = ["___v4", "___v0.scaleai", "", "___metadata"]
        if len(ann)>0:
            img_ann.append((p, ann))
    return img_ann

def get_annotation(ann_files):
    """ Finding annotation file and read content
        @param: ann_files - Paths to annotation .json files
        @return: annotations
    """
    def _load(f):
        with open(str(f), 'r') as of:
            return json.load(of)
    
    def _extract_text(text):
        if text and '_' in text:
            tt = [t for t in text.split('_') if t]
            text = tt[0] if tt else None
        return text
    
    def _clean_text(text):
        if text:
            text = alphabet.clean(text)
        if text:
            text = alphabet.take_first_uncertain(text)
        return text if text else None
    
    def _state_name_to_canonic(state):
        if state:
            state = state.strip().replace(' ', '-').split('_')[0]
        if state:
            state = geo_names.canonic_state_name(state)
        return state if state else None

    def _get_meta_info(meta):
        names = [
            'axonai.alpr.annotation.block',
            #'axonai.alpr.annotation.camera.position',
            #'axonai.alpr.annotation.fw',
            'axonai.alpr.annotation.lighting.condition',
            'axonai.alpr.annotation.scenario',
            'axonai.alpr.annotation.speed',
            'axonai.alpr.annotation.state'
        ]
        info  = {}
        for n in names:
            n_ = n.split('.')[-1]
            v  = utils.find_value(meta, n)
            if n_ == 'block':
                info[n_] = (0 if not v else int(v))
            else:
                if n_ == 'state':
                    n_ = 'location'
                info[n_] = ('' if not v else v)
        return info

    def _get_jsondata(ann):
        ann_ = []
        try:
            for a in ann:
                if isinstance(a, dict):
                    ann_ += a.get('jsondata', [])
        except Exception:
            pass
        return ann_ if ann_ else ann
    
    def _get_ann(ann):
        ann_plates = []
        ann = [a for a in ann if isinstance(a, dict)]
        # If data are wrapped        
        ann = _get_jsondata(ann)
        # Get annotation: plate texts, ...
        for a in ann:
            # Get plate text
            text_ann = utils.find_value(a, 'licensePlateText')
            text_orig = _extract_text(text_ann)
            text = _clean_text(text_orig)
            # Get plate coords
            coord = utils.find_value(a, 'licensePlateLocation')
            # Get US State name
            state_ann = utils.find_value(a, 'licensePlateState')
            state = _state_name_to_canonic(state_ann)

            if text_ann or coord or state_ann:
                ann_plates.append({
                    'text_ann': (clean_str(text_ann) if text_ann else ''),
                    'text': (text if text else ''),
                    'is_stacked': alphabet.is_stacked(text_orig),
                    'text_len': alphabet.get_len(text_orig),
                    'coord': coord,
                    'state_ann': (clean_str(state_ann) if state_ann else ''),
                    'state': (state if state else ''),
                    'state_fixed': False
                })
        return ann_plates

    ann_status, ann_meta, ann_plates = None, None,  []
    for f in ann_files:
        stem = Path(f).stem
        if 'qualityreport' in stem:
            try:
                ann = _load(f)
                ann_status = utils.find_value(ann, 'quality_status')
            except Exception:
                pass
        elif 'metadata' in stem:
            try:
                ann = _load(f)
                ann_meta = _get_meta_info(ann)
            except Exception:
                pass
        elif not ann_plates:
            try:
                ann = _load(f)
                ann_plates = _get_ann(ann)
            except Exception:
                pass
    return ann_status, ann_meta, ann_plates

def topleft_topright_botright_botleft(coord):
    """ Reordering of annotated coordinates of plate vertices.
        @param: coord - Annotated coordinates
        @return: plate points in clockwise order:
            1----------2
            |          |
            4----------3
    """
    try:
        x1, y1 = coord[0]['x'], coord[0]['y']
        x2, y2 = coord[1]['x'], coord[1]['y']
        x3, y3 = coord[2]['x'], coord[2]['y']
        x4, y4 = coord[3]['x'], coord[3]['y']

        # Ordering of coordinate bounds
        left  = min([x1, x2, x3, x4])
        right = max([x1, x2, x3, x4])
        top   = min([y1, y2, y3, y4])
        bot   = max([y1, y2, y3, y4])

        def _find_argmin(x_, y_):
            """Find the index of closest point"""
            d1 = (x1 - x_) ** 2 + (y1 - y_) ** 2
            d2 = (x2 - x_) ** 2 + (y2 - y_) ** 2
            d3 = (x3 - x_) ** 2 + (y3 - y_) ** 2
            d4 = (x4 - x_) ** 2 + (y4 - y_) ** 2
            return np.argmin([d1, d2, d3, d4])

        _1 = _find_argmin(left, top)
        _2 = _find_argmin(right, top)
        _3 = _find_argmin(right, bot)
        _4 = _find_argmin(left, bot)

        x1, y1 = coord[_1]['x'], coord[_1]['y']
        x2, y2 = coord[_2]['x'], coord[_2]['y']
        x3, y3 = coord[_3]['x'], coord[_3]['y']
        x4, y4 = coord[_4]['x'], coord[_4]['y']

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    except Exception:
        return None

def extend_plate_bounds(points, h, w, extend_factor):
    """Extend borders of rectangular plate
        @param: points - top-left, top-right, bottom-right, bottom-left
        @param: h, w - Frame image height and width
        @return: wider bounds: top-left, top-right, bottom-right, bottom-left
    """
    if extend_factor > 0:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]

        dx12 = (x2 - x1) * extend_factor
        dy14 = (y4 - y1) * extend_factor
        dx34 = (x3 - x4) * extend_factor
        dy23 = (y3 - y2) * extend_factor

        x1 = max([  0, x1 - dx12])
        x2 = min([w-1, x2 + dx12])
        x3 = min([w-1, x3 + dx34])
        x4 = max([  0, x4 - dx34])

        y1 = max([  0, y1 - dy14])
        y2 = max([  0, y2 - dy23])
        y3 = min([h-1, y3 + dy23])
        y4 = min([h-1, y4 + dy14])

        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return points

def calc_area(points):
    edges = zip(points, points[1:] + [points[0]])
    area = np.sqrt(0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in edges)))
    return area

def calc_h_w(points):
    left  = points[:,0].min()
    right = points[:,0].max()
    top   = points[:,1].min()
    bot   = points[:,1].max()
    h = bot - top
    w = right - left
    return h, w

def calc_area_score(points):
    """ Calc area_score.
            @param: points - Coordinates of plate vertices
            @return: area_score in range [0.0; 1.0]
    @return:
    """
    area = calc_area(points)
    min_area = np.sqrt(18 * 27)
    max_area = 5 * min_area
    if area <= min_area:
        return 0.0
    elif area >= max_area:
        return 1.0
    return (area - min_area) / (max_area - min_area)

def extract_annotation(img_path, ann_files, log):
    """ Extract plates.
        @param: img_path - Path to input frame image
        @param: ann_files - List of annotation .json-file paths
        @param: log - List of all extraction logging
        @return: None
    """
    row = LOG_DEFAULTS.copy()
    row[LOG_IDX['frame_id']] = Path(img_path).stem
    row[LOG_IDX['id']] = uuid.uuid1()
    row[LOG_IDX['frame_file']] = str(img_path)
    try:
        ann_status, ann_meta, ann_plates = get_annotation(ann_files)
        
        add_val(row, 'ann_status', ann_status)
        for c in ['block', 'condition', 'scenario', 'speed','location']:
            add_val(row, c, ann_meta.get(c, ''))
        
        if len(ann_plates)>0:
            for a in ann_plates:
                row_ = row.copy()
                row_[LOG_IDX['id']] = uuid.uuid1()

                for c in ['text_ann', 'text', 'text_len', 'is_stacked', 'state_ann', 'state', 'state_fixed']:
                    add_val(row_, c, a.get(c, None))

                points = topleft_topright_botright_botleft(a.get('coord', None))
                if points is None:
                    add_error_msg(row, "No coords")
                else:
                    row_[LOG_IDX['x1_ann']], row_[LOG_IDX['y1_ann']] = points[0]
                    row_[LOG_IDX['x2_ann']], row_[LOG_IDX['y2_ann']] = points[1]
                    row_[LOG_IDX['x3_ann']], row_[LOG_IDX['y3_ann']] = points[2]
                    row_[LOG_IDX['x4_ann']], row_[LOG_IDX['y4_ann']] = points[3]

                log.append(row_)
        else:
            add_error_msg(row, "No annotation")
            log.append(row)

    except Exception as e:
        add_error_msg(row, str(e) + ' - ' + clean_str(str(traceback.format_exc())))
        log.append(row)

def correction_of_state_names(log):
    """ Correction of State names grouping data by plate texts
        @param: log - Extraction log
        @return: Restore missed/incorrect US State names in log.
    """
    # Building a map: text => list(states)
    plates = {}
    t, s = LOG_IDX['text'], LOG_IDX['state']
    for row in log:
        text, state = row[t], row[s]
        if text and state:
            if text not in plates:
                plates[text] = []
            plates[text].append(state)
    # Finding most likely state for each plate with text
    plates = {text: utils.most_likely_state(states) for text, states in plates.items()}
    # Filter off empty states
    plates = {text: state for text, state in plates.items() if state}
    # Fixed US State names
    stat = []
    x = LOG_IDX['state_fixed']
    for row in log:
        row[x] = False
        correct_state = plates.get(row[t], '')
        if correct_state and row[s] != correct_state:
            stat.append(f"'{row[s]}' to '{correct_state}'")
            row[s] = correct_state
            row[x] = True
    if stat:
        print("\nUS States renaming statistics:")
        print(f"{'State':40s}\t{'Count':6s}")
        for p in Counter(stat).items():
            print(f"{str(p[0]):40s}\t{p[1]:6d}")

def extract_plate(row, out_dir, debug_dir, min_score=0.1, extend_factor=0.025):
    """ Extract plates.
        @param: row - Record in log
        @param: out_dir - Directory to save extracted plate images
        @param: debug_dir - Directory to save plate images with errors
        @param: min_score - Min value for plate image score
        @param: extend_factor - Extend factor for plate rectangular bounds
        @return: None
    """
    try:
        frame_path = row[LOG_IDX['frame_file']]
        img = imgtf.imread(frame_path)

        if img is None:
            raise Exception("No frame image")
        if img.shape[0]==0 or img.shape[1]==0:
            raise Exception("Corrupted frame image")

        points = np.array([
            [row[LOG_IDX['x1_ann']], row[LOG_IDX['y1_ann']]], 
            [row[LOG_IDX['x2_ann']], row[LOG_IDX['y2_ann']]], 
            [row[LOG_IDX['x3_ann']], row[LOG_IDX['y3_ann']]], 
            [row[LOG_IDX['x4_ann']], row[LOG_IDX['y4_ann']]]
        ])

        if points.mean()==0:
            raise Exception("Wrong coordinates")

        # Extention of plate area
        h, w = img.shape[0], img.shape[1]
        points = extend_plate_bounds(points, h, w, extend_factor)

        row[LOG_IDX['area']] = calc_area(points)
        row[LOG_IDX['h']], row[LOG_IDX['w']] = calc_h_w(points)
        row[LOG_IDX['x1']], row[LOG_IDX['y1']] = points[0]
        row[LOG_IDX['x2']], row[LOG_IDX['y2']] = points[1]
        row[LOG_IDX['x3']], row[LOG_IDX['y3']] = points[2]
        row[LOG_IDX['x4']], row[LOG_IDX['y4']] = points[3]
        area_score = row[LOG_IDX['area_score']] = calc_area_score(points)

        # Extract plate image from frame
        plate_img = imgtf.warp(img, points)
        if plate_img is None or plate_img.shape[0]==0 or plate_img.shape[1]==0:
            raise Exception("Warp failed")

        row[LOG_IDX['pixels_score']] = imgtf.calc_pixels_score(plate_img)
        blur_score = row[LOG_IDX['blur_score']] = imgtf.calc_blurring_score(plate_img)
        row[LOG_IDX['score']] = area_score * blur_score

        score_ = blur_score # score

        text = row[LOG_IDX['text']]
        block = row[LOG_IDX['block']]
        ann_status = row[LOG_IDX['ann_status']]

        flag_ok = text and score_ >= min_score # and ann_status in ANN_STATUS['passed']
        
        dir1 = 'text_ok' if text else 'text_empty'
        dir2 = 'score_ok' if score_ >= min_score else 'score_0' if score_ == 0 else 'score_low'
        dir3 = 'ann_passed' if ann_status in ANN_STATUS['passed'] or block == 0 else 'ann_failed' if ann_status in ANN_STATUS['failed'] else 'ann_pending'

        row[LOG_IDX['dir']] = out_dir.name if flag_ok else debug_dir.name if debug_dir else ''
        row[LOG_IDX['subdir']] = '' if flag_ok else '/'.join([dir1, dir2, dir3])

        if flag_ok or debug_dir:

            info = {'label': text, 'state': row[LOG_IDX['state']], 'block': block, 'score': score_, 'hash': row[LOG_IDX['frame_id']], 'idx': '0'}

            out_dir_ = out_dir if flag_ok else utils.mk_dir(debug_dir.joinpath(dir1, dir2, dir3))
            plate_path = utils.get_not_existed_path(out_dir_, info)
            
            is_saved = imgtf.imwrite(plate_img, plate_path)
            if is_saved:
                row[LOG_IDX['file']] = str(plate_path.name)
            else:
                add_error_msg(row, "Imwrite failed")

    except Exception as e:
        add_error_msg(row, str(e))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',         type=str,   required=True,     help="Source of frames (.jpg) and annotattions (.json). 1) AzureML Dataset 'NAME:VERSION' or 2) Input dir. path")
    parser.add_argument('--outdir',        type=str,   required=True,     help="Output directory with plates image files" )
    parser.add_argument('--debugdir',      type=str,   required=False,    help="Output directory with plates image files with errors" )
    parser.add_argument('--clear_outdir',  type=int,   default=0,         help='Clear output directory')
    parser.add_argument('--logdir',        type=str,   default='outputs', help="Output directory with logging info." )
    parser.add_argument('--min_score',     type=float, default=0.0,       help='Min value for score threshold (but not 0).')
    parser.add_argument('--extend_factor', type=float, default=0.025,     help="Extend factor for plate image.")
    parser.add_argument('--process_count', type=int,   default=8,         help="Number of processes")
    args = parser.parse_args()

    print(f"Start plate images extraction. Time: {str(datetime.now())}")
    start_time = time()
    utils.print_args(args)
    print(f"Start extraction of plate images.")
    now_date = str(datetime.now().date())
        
    out_dir   = utils.mk_or_clear_dir(args.outdir,   bool(args.clear_outdir))
    debug_dir = utils.mk_or_clear_dir(args.debugdir, bool(args.clear_outdir)) if args.debugdir else None
    log_dir   = utils.mk_dir(args.logdir)


    # Get files: frames (.jpg) and annotattions (.json)
    print(f"Get files from {args.indir}")
    indir, dataset, mounted = None, None, None
    img_files, ann_files = [], []
    tag = ''
    if ONLINE:
        n, v = args.indir.split(':')
        tag = n
        dataset = Dataset.get_by_name(WORKSPACE, name=n, version=int(v))
        mounted = dataset.mount()
        print(f"Mounted point: {mounted.mount_point}")
        indir = Path(mounted.mount_point)
        for f in dataset.to_path():
            p = Path(mounted.mount_point + f)
            if p.suffix == '.jpg':
                img_files.append(p)
            elif p.suffix == '.json':
                ann_files.append(p)        
    else:
        indir = Path(args.indir)
        tag = indir.name
        img_files = list(indir.rglob(f'*.jpg'))
        ann_files = list(indir.rglob(f'*.json'))
    print(f"Number of frames: {len(img_files)}")    
    
    
    # Grouping annotation files by frames
    img_ann = group_annotations_by_frames(img_files, ann_files)
    assert img_ann, f"No input data."
    print(f"Input frames: {len(img_ann)}")

    if ONLINE:
        mounted.start()

    print("Extraction of  annotations ...")
    log = []
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(extract_annotation)(img_path, ann_files, log)
        for img_path, ann_files in tqdm(img_ann, total=len(img_ann), mininterval=max(len(img_ann)//500, 1))
    )

    print("Correction of State names grouping data by plate texts ...")
    correction_of_state_names(log)


    print("Extraction of plate images ...")
    Parallel(n_jobs=args.process_count, prefer='threads')(
        delayed(extract_plate)(row, out_dir, debug_dir, args.min_score, args.extend_factor)
        for row in tqdm(log, total=len(log), mininterval=max(len(log)//500, 1))
    )
    
    if ONLINE:
        mounted.stop()

    df = save_log(log, log_dir.joinpath(f"extract_log_{tag}_{now_date}.csv"), indir)

    # Logging
    t, f, e = LOG_IDX['text'], LOG_IDX['file'], LOG_IDX['err_msg']
    files = [row[f] for row in log if row[f] and row[t] and not row[e]]
    num_extracted = len(files)
    print(f"\nNumber of extracted plates: {num_extracted}")
    num_errors = sum([1 for row in log if row[e]])
    num_total = len(log)
    print(f"Errors: {num_errors:d} ({100*num_errors/num_total:.2f}%) of total {num_total:d}")

    print("\n-------------------------------------------")
    print("Statistics of extracted plates with text")
    print("-------------------------------------------\n")
    utils.group_by_plate_text(files, min_score=0, show=True)

    assert num_extracted > 0.1 * num_total, "Error! Number of extracted data is less than 10%"

    print(f"Elapsed time: {utils.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
