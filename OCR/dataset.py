#!/usr/bin/python
# encoding: utf-8

import random
import sys
from pathlib import Path
import lmdb
import six
from PIL import Image # Pillow
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None, rgb=False, test_mode=False):
        path = Path(root).resolve()
        for f in ['data.mdb', 'lock.mdb']:
            assert path.joinpath(f).exists(), f"Missed file {f} in LMDB dataset: {root}"

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        assert self.env, f"Cannot creat lmdb from: {root}"

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode('utf-8')))
        assert self.nSamples > 0, f"No samples found in: {root}"

        self.transform = transform
        self.target_transform = target_transform
        self.rgb = rgb
        self.test_mode = test_mode

    def __len__(self):
        return self.nSamples

    def _get_value(self, txn, name, key, default=None, do_decode=False):
        value  = txn.get(f"{name}-{key}".encode('utf-8'))
        if value is None:
            return default
        return value.decode(encoding='UTF-8') if do_decode else value

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        key = f"{index+1:09d}" # 1-based numeration
        
        with self.env.begin(write=False) as txn:

            imgbuf = self._get_value(txn, 'image', key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:                
                img = Image.open(buf)
                if not self.rgb:
                    img = img.convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label = self._get_value(txn, 'label', key)
            if self.target_transform is not None:
                label = self.target_transform(label)

            state = self._get_value(txn, 'state', key, default="", do_decode=True)

            if self.test_mode:
                hash_id = self._get_value(txn, 'hash', key, default="", do_decode=True)
                block   = self._get_value(txn, 'block', key, default=0, do_decode=True)
                block = int(block) if block else 0
                score   = self._get_value(txn, 'score', key, default=1, do_decode=True)
                score = float(score) if score else 0
                return (img, label, state, hash_id, block, score)
            else:
                return (img, label, state)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels, states = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
