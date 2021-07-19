#!/usr/bin/python
# encoding: utf-8

from typing import List, Union, Optional
import os
import re
import csv
from difflib import SequenceMatcher
from pathlib import Path
import shutil
import numpy as np
import cv2
import yaml

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import alphabets
families = alphabets.alphabet

def to_bool(val):
    """ Command line argument to bool.
            @param: val - One of True, False, T, F, 1, 0
            @return: True or False
    """
    return str(val).lower() in ['1','true','t']

def select_device(device: str = 'gpu', batch_size: int  = None):
    """ Select device: cpu or gpu (max available gpu nodes) or '0,1,2,3' (selected nodes). Set var CUDA_VISIBLE_DEVICES.
            @param: device = 'cpu' or 'gpu' or '0' or '1' or '0,1,2,3' ...
            @param: batch_size - batch size (should be multiple of GPU count)
            @return: torch device type - 'cpu' or 'cuda:0'
    """
    cuda =  None
    device = device.lower()
    s = f'torch {torch.__version__} '
    ptrn = re.compile(r"^(gpu|\d(,\d)*)$")
    if bool(ptrn.match(device)) and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if device == 'gpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(n)))
        else:
            _n = len(device.split(','))
            assert _n <= n, f"Error! Available CUDA devices: {n} but requested: {_n} in --device {device}"
            n = _n
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'Error! --batch_size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(range(n) if device == 'gpu' else device.split(',')):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i==0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
        cuda  = torch.device('cuda:0')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        s += 'CPU\n'
        cuda  = torch.device('cpu')
    print(s)
    return cuda.type

def clear_dir(dir_path):
    """ Create or clear dir.
        If not exists - create, otherwise - remove all content: files and subdirs.
        @return: Path(dir_path)
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    elif dir_path.is_dir():
        for f in dir_path.glob("*"):
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(str(f))
    return dir_path

def error_save(errors, target, pred, image, hash_id, path):
    """ Saving images with errors to directory for error analysis.
            @param: errors  - list of tuples: (groundtruth, prediction)
            @param: target  - groundtruth plate text
            @param: pred    - prediction plate text
            @param: image   - plate image to save
            @param: hash_id - id of frame
            @param: path    - output falder
            @return: None 
    """
    errors.append((target, pred))
    img = image.squeeze(1).permute(1,2,0).cpu().numpy()
    img = np.array(img*255+128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    file_path = Path(path).joinpath(f"incorrect_{target}_{pred}_{hash_id}.jpg")
    cv2.imwrite(str(file_path), img)

def get_source_root(indir, names):
    """ Get whole path to source data dir with .mdb files.
            @param: indir - input directory
            @param: names - list or comma sep str of names 
            @return the whole Path to training data
        Example:
            indir = "data"
            names = "test,valid"
            Files structure:
                data
                    /blah
                        /blah
                            /valid/*.mdb <- this will be found
                    /blah
                        /blah
                            /train/*.mdb
            return Path("data/blah/blah/valid/")
    """
    if not indir:
        return None
    p = Path(indir).resolve()
    # If data.mdb file is located directly in indir, return indir.
    if list(p.glob('data.mdb')):
        return p
    # But if data.mdb file is located in one of multiple sub-dirs of indir, try to find it.
    # Find all paths to directories which contain data.mdb file.
    dd = set([str(f.parent) for f in p.rglob('data.mdb')])
    # Among all paths to data.mdb files select these which contains specific names: train, test, valid, ...
    if isinstance(names, str):
        names = names.split(',')
    dd = list(filter(lambda d: len([n for n in names if n in d.lower()])>0, dd))
    # We may still have multiple paths and only user can resolve the ambiguity.
    assert len(dd)<=1, f"Error! Define only one source of: {', '.join(dd)}"
    # Success is if we found exactly one path.
    p = Path(dd[0]) if len(dd)==1 else None
    return p if p and p.exists() else None

def write_log(file, row, mode='w', delim=','):
    """ Logging data to .csv file.
            @param: file  - file path
            @param: row   - list of data to save into file
            @param: mode  - 'r' (read), 'w' (write), 'a' (append)
            @param: delim - delimiter
            @return: None
    """
    with open(str(file), mode, newline='') as f:
        writer = csv.writer(f, delimiter=delim, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)

def optimizer_build(model, params, lr=None):
    """ Building training optimizer.
            @param: model  - training model
            @param: params - parameters
            @param: lr     - learning rate
            @return: optimizer (Adam/Adadelta/RMSprop)
    """
    if lr is None:
        lr = params.lr
    weight_decay = params.weight_decay if hasattr(params, 'weight_decay') else 0
    if params.adam:
        return optim.Adam(model.parameters(), lr=lr, betas=(params.beta1, 0.999), weight_decay=weight_decay)
    elif params.adadelta:
        return optim.Adadelta(model.parameters(), weight_decay=weight_decay)
    return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

def match_seq(list1, list2):
    """A generator that yields matches of list1 vs list2"""
    s = SequenceMatcher(None, list1, list2)
    for block in s.get_matching_blocks():
        for i in range(block.size):
            yield block.a + i, block.b + i # you don't need to store the matches, just yields them

def create_convert(*families):
    """Return a converter function that converts a list
    to the same list with only main words"""
    d = {w:main for main, *alternatives in map(list, families) for w in alternatives}
    return lambda L: [d.get(w, w) for w in L]

convert = create_convert(*families)

def list_compare(list1, list2):
    
    match = []
    for a,b in match_seq(convert(list1), convert(list2)):
        match.append([a,b, list1[a],list2[b]])
        print([a,b, list1[a],list2[b]])
    
    list1_match = [_[0] for _ in match]
    list2_match = [_[1] for _ in match]
    
    miss_read_target = []
    miss_read_pred = []
    confuse_list = []
    for i in range(len(list1)):
        if i not in list1_match:
            miss_read_target.append(list1[i])
            
    for i in range(len(list2)):
        if i not in list2_match:
            miss_read_pred.append(list2[i])
            
    if len(list1) == len(list2):
        confuse_list += [(list1[i],list2[i]) for i in range(len(list1)) if list1[i] != list2[i]]
        
    return miss_read_target, miss_read_pred, confuse_list

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
    """

    def __init__(self, alphabet, device):
        self.device = device # 'cpu' or 'cuda'
        self.alphabet = alphabet + '-'  # for `-1` index
        # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        self.dict = {char: i + 1 for i, char in enumerate(self.alphabet)}
    
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """
        result, length = [], []
        for item in text:            
            item = item.decode('utf-8','strict').upper()
            r = [self.dict[c] for c in item if c in self.dict]
            length.append(len(r))
            result.append(r)
        
        max_len = max(length)
        result = [r + [0] * (max_len - len(r)) for r in result]
        return (torch.LongTensor(result).to(self.device), torch.LongTensor(length).to(self.device))

    def decode(self, t, length, raw=False, always_list=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                res = ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                res = ''.join(char_list)

            if always_list:
                return [res]
            else:
                return res

        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts

    def decode_2heads(self, t, t_s,length, raw=False):
            """Decode encoded texts back into strs.

            Args:
                torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
                torch.LongTensor [n]: length of each text.

            Raises:
                AssertionError: when the texts and its length does not match.

            Returns:
                text (str or list of str): texts to convert.
            """
            if length.numel() == 1:
                fst = self.decode(t,length,raw= True)
                sec = self.decode(t_s,length,raw= True)

                text = ''.join([fst[i] if sec[i] == '-' else sec[i] for i in range(len(fst))])
                if raw:
                    return text
                else:
                    char_list = []
                    for i in range(length):
                        if text[i] != '-' and (not (i > 0 and text[i - 1] == text[i])):
                            char_list.append(text[i])
                    return ''.join(char_list)

            else:
                # batch mode
                assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                    t.numel(), length.sum())
                texts = []
                index = 0
                for i in range(length.numel()):
                    l = length[i]
                    texts.append(
                        self.decode_2heads(
                            t[index:index + l], t_s[index:index + l], torch.LongTensor([l]), raw=raw))
                    index += l
                return texts

def get_item(val):
    if isinstance(val, (Variable, torch.Tensor)):
        return val.item()
    return val

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset averager to initial state.
        """
        self.sum = 0
        self.count = 0

    def add(self, val):
        """ Add value to sum and increment total count by 1.
            @param: val - adding value
            @return: None
        """
        count = 1
        if isinstance(val, Variable):
            val = val.data.sum()
            count = val.data.numel()
        elif isinstance(val, torch.Tensor):
            val = val.sum()
            count = val.numel()
        self.sum += val
        self.count += count

    def val(self):
        """ Calculate average value for accumulated values.
            @return: mean
        """
        return self.sum / float(self.count) if self.count != 0 else self.sum

def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)

def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))

def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

def no_stack(name):
    """ Delete square bracket blocks with two or more symbols. Delete all square brackets after that.
    Example: AB[CD]1[2] -> AB12
    """
    p = re.compile(r"(\[[^\[\]]{2,}\])")
    name = p.sub('', name)
    name = name.replace('[','').replace(']','')
    return name.encode('utf-8')

def no_stack_2(name):
    """ Delete all square bracket blocks.
    Example: AB[CD]1[2] -> AB1
    """
    p = re.compile(r"(\[[^\[\]]{1,}\])")
    name = p.sub('', name)
    name = name.replace('[','').replace(']','')
    return name.encode('utf-8')

def one_line(name):
    """ Delete all square brackets.
    Example: AB[CD]1[2] -> ABCD12
    """
    name = re.sub(r"[\[\]\(\)\{\}]", '', name)
    return name.encode('utf-8')

def one_line_2(name):
    """ Extract content of square bracket blocks only.
    Example: AB[CD]1[2] -> CD2
    """
    p = re.compile(r"\[([^\[\]]{1,})\]")
    name = ''.join(p.findall(name))
    return name.encode('utf-8')

def one_line_3(name):
    """ Extract content of square bracket blocks with two or more symbols.
    Example: AB[CD]1[2] -> CD
    """
    p = re.compile(r"\[([^\[\]]{2,})\]")
    name = ''.join(p.findall(name))
    return name.encode('utf-8')

def two_lines(name):
    """ Delete all square bracket blocks. Extract content of square bracket blocks only.
    Example: AB[CD]1[2] -> (AB12, ABCD12)
    """
    return (no_stack_2(name), one_line_2(name))

def take_first_option(name):
    """ Take first option from block <...>.
    Example: AB<D0>1<2Z7> -> ABD12
    """
    options = re.findall(r"(<[^<>]*>)", name)
    for opt in options:
        o = opt[1] if len(opt) > 2 else ''
        name = name.replace(opt, o)
    return name

def text_transform(texts, label):
    """ Plate text transformation.
            @param: texts - plate text or list of texts
            @param: label - 'no-stack'/'one-line'/'classify'/'two-lines'
            @return: tuple: (original texts, transformed texts)
    """
    original_texts = [text.decode('utf-8', 'strict') for text in texts]
    texts = [take_first_option(text) for text in original_texts]
    if label == 'no-stack':
        texts = [no_stack(text) for text in texts]
    elif label == 'one-line' or label == 'classify':
        texts = [one_line(text) for text in texts]
    elif label == 'two-lines':
        texts = [two_lines(text) for text in texts]
    return (original_texts, texts)

class LearningRateScheduler():
    """ 
    """
    def __init__(self, lr, lr_period, min_lr, epochs):
        self._lr = lr
        self._lr_period = lr_period
        self._min_lr = min_lr
        self._epochs = epochs
        self._patient = 0
        self._patient_milestone = lr_period
        self._best_accuracy = 0
        self._best_epoch = 0
        self._best_batch = 0
        self._stop_training = False

    @property
    def lr(self):
        return self._lr

    @property
    def stop_training(self):
        return self._stop_training
    
    @property
    def best_accuracy(self):
        return self._best_accuracy

    @property
    def best_accuracy_str(self):
        return f"{self._best_accuracy*100:.2f}%"

    def _reset_learning_rate(self):
        """ Reduce learning rate.
        """
        self._lr = self._lr * 0.1 ** 0.5
    
    def _reset_accuracy(self, accuracy, epoch=0, batch=0):
        """ Set a new best accuracy and reset other parameters.
                @param: accuracy - New accuracy
                @param: epoch - Current epoch number
                @param: batch - Current batch number
        """
        self._best_accuracy = accuracy
        self._best_epoch = epoch
        self._best_batch = batch
        self._patient = 0
        self._patient_milestone = self._lr_period
    
    def _stop(self, msg=''):
        """ Set stop training flag as True
            @param: msg - Message with a reson of stopping training.
        """
        self._stop_training = True
        print(f"Stop training! {msg}")

    def update(self, accuracy, epoch=0, batch=0):
        """ Decide what to do next:
            1) stop training
            2) update accuracy and learning rate
            3) waite more batches to select between 1) and 2)
                @param: accuracy - New accuracy
                @param: epoch - Current epoch number
                @param: batch - Current batch number

        """
        if epoch >= self._epochs:
            self._stop(msg='Reached max number of epochs.')
            return False          
        
        if accuracy > self._best_accuracy:
            self._reset_accuracy(accuracy, epoch, batch)
            return True

        self._patient += 1
        print("Accuracy not increase in {:d} intervals. Best accuracy: {:.4f} at epoch {:d} and batch {:d}. Learning rate {:.8f}".format(
            self._patient, self._best_accuracy, self._best_epoch, self._best_batch, self._lr))
        if self._patient >= self._patient_milestone:
            self._reset_learning_rate()
            if self._lr < self._min_lr:
                self._stop(msg='LR too small.')
            else:
                self._patient_milestone += self._lr_period
        return False
