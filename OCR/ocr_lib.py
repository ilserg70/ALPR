#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import datetime
from time import time
import importlib
import shutil
import re
import csv
import random
from difflib import SequenceMatcher
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torch.nn import CTCLoss
cudnn.benchmark = True
# Import horovod lib (for distributed claud calculation)
# Horovod Installation Guide: https://horovod.readthedocs.io/en/stable/install_include.html
# Build a Conda Environment with GPU Support for Horovod: https://horovod.readthedocs.io/en/stable/conda.html
import horovod.torch as hvd

from tool import utils
import dataset
from layers import print_model_info
from azureml_jobs import azureml_utils as aml

if not aml.ONLINE:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(write_to_disk=False)

ONLINE = aml.ONLINE
IS_HVD = False

US_STATES = [
    'Alabama',
    'Alaska',
    'Arizona',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut',
    'Delaware',
    'District-Of-Columbia',
    'Florida',
    'Georgia',
    'Hawaii',
    'Idaho',
    'Illinois',
    'Indiana',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Massachusetts',
    'Michigan',
    'Minnesota',
    'Mississippi',
    'Missouri',
    'Montana',
    'Nebraska',
    'Nevada',
    'New-Hampshire',
    'New-Jersey',
    'New-Mexico',
    'New-York',
    'North-Carolina',
    'North-Dakota',
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Rhode-Island',
    'South-Carolina',
    'South-Dakota',
    'Tennessee',
    'Texas',
    'Utah',
    'Vermont',
    'Virginia',
    'Washington',
    'West-Virginia',
    'Wisconsin',
    'Wyoming']
BLOCKS = list(range(10))
SCORES = [round(i/10,1) for i in range(0,11)]

def activate_hvd(is_hvd):
    global IS_HVD
    IS_HVD = is_hvd and ONLINE
    if IS_HVD:
        hvd.init()
        print(f"Init horovod: hvd.rank = {hvd.rank()}, hvd.local_rank = {hvd.local_rank()}, hvd.size = {hvd.size()}")
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())

def print_args(args):
    print("Input arguments:")
    for arg in vars(args):
        print("{0:20s} {1:}".format(arg, getattr(args, arg)))

def to_bool(val):
    """ Command line argument to bool.
            @param: val - One of True, False, T, F, 1, 0
            @return: True or False
    """
    return str(val).lower() in ['1','true','t']

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

def clear_dir(dir_path):
    """ Create or clear dir.
        If not exists - create, otherwise - remove all content: files and subdirs.
        @return: Path(dir_path)
    """
    if not dir_path:
        return None
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

def set_random_seed(seed, device):
    """ Set random seed. Ensure everytime the random is the same
            @param: seed
            @param: device - cpu or cuda
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed(seed)

def select_device(device, batch_size):
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

def import_net_module(module_name):
    """ Import model py-module and do model initialization.
            @param: module_name - Net module name or path
            @return: net_module, net_type
    """
    net_type = Path(module_name).stem
    p = Path('models').joinpath(f"{net_type}.py")
    assert module_name and p.exists() and p.is_file(), f"Error! Net module {module_name} doesn't exist."

    net_module = importlib.import_module(f"{p.parent.name}.{p.stem}")
    if not hasattr(net_module, 'label'):
        net_module.label = 'no-stack'
        print("Label type:", net_module.label)
    if not hasattr(net_module, 'weight_decay'):
        net_module.weight_decay = 0
        print('Weight decay:', net_module.weight_decay)
    return net_module, net_type

def init_model(net_module):
    """ Import model py-module and do model initialization.
            @param: net_module - Net module
            @return: model
    """
    model = net_module.OCR(imgH=net_module.imgH, nc=net_module.nc, nclass=len(net_module.alphabet)+1, nh=net_module.nh)
    print_model_info(model, net_module, is_layers=False) # This code should be executed before parallelization (cuda)    
    return model

def prunning(model, net_module):
    """ Model prunning and print info. 
        Call: import_net_module(..) and init_model(..) and then this func.
        Note: Don't call model_prepare(..) with 'cuda' or IS_HVD=True
            @param: model - Model object
            @param: net_module - Net module
            @return None
    """
    print("Start model prunning.")
    model.eval()
    x = torch.randn(1, net_module.nc, net_module.imgH, net_module.imgW)
    from tensorly.decomposition import tucker, parafac
    for module in [model.cnn, model.cnn1, model.cnn2, model.cnn3]:
        submodules = [submodule for submodule in module.modules() if type(submodule) == torch.nn.Conv2d]
        for i,submodule in enumerate(submodules):
            m = submodule.weight.data.numpy()
            core, factors = tucker(m,rank=[m.shape[0],m.shape[1],1,1])
            m = core.squeeze(2).squeeze(2)
            _, s, _ = np.linalg.svd(m.dot(m.transpose()))
            s = sorted(s)
            if i == 0: s = s[:-3]
            print(m.shape[0],len([ num for num in s if num > max(s)/10 ]))# m = np.reshape(m,(48,27))

def load_model_weights(model, weights_source, device):
    """ Loading model weights.
            @param: model - Model object
            @param: weights_source - Path to .pth file or registered model name on AzureML
            @param: device - cpu or gpu
            @return: True if weights are loaded
    """
    if weights_source:
        print(f"Loading model weights from: {str(weights_source)}")
        p = Path(weights_source)
        if not p.exists() and ONLINE:
            p = Path(aml.get_model(weights_source))
        assert p.exists() and p.is_file(), f"Error! Failed downloading model weights from: {weights_source}"
        dict_to_load = torch.load(str(p), map_location=torch.device(device))
        # Removing prefix 'module.' from dictionary keys.
        dict_to_load = {key.replace('module.', ''): val for key, val in dict_to_load.items()}
        model.load_state_dict(dict_to_load)

def model_prepare(model, net_module, device):
    """ Prepare model for training/validation.
            @param: model - Model object
            @param: net_module - Net module
            @param: device - cpu or gpu
            @return: model - transformed model object 
    """
    model.eval()
    if net_module.dealwith_lossnan and torch.__version__ < '1.1.0':
        # We use this way to change the inf to zero
        model.register_backward_hook(model.backward_hook)
    if device=='cuda':
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    if IS_HVD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    return model

def get_converter(net_module, device):
    """ Convert between str and label.
    """
    return utils.strLabelConverter(net_module.alphabet, device)

def criterion_init(net_module, focal_loss, device):
    """ Initialization of training criterion.
            @param: net_module  - Net module
            @param: focal_loss
            @param: device - cpu or cuda
            @return: criterion
    """
    reduction_loss = 'none' if to_bool(focal_loss) else 'mean'
    criterion = CTCLoss(reduction = reduction_loss)
    if net_module.dealwith_lossnan and torch.__version__ >= '1.1.0':
        # zero_infinity - whether to zero infinite losses and the associated gradients.
        # Infinite losses mainly occur when the inputs are too short to be aligned to the targets.
        criterion = CTCLoss(zero_infinity = True, reduction = reduction_loss)
    if device=='cuda':
        criterion = criterion.to(device)
    return criterion

def criterion1_init():
    return torch.nn.BCEWithLogitsLoss()

def if_focal_loss(cost, focal_loss, gamma):
    """ Transform training cost if focal loss is defined.
            @param: cost - cost value
            @param: focal_loss
            @param: gamma
            @return: transformed cost
    """
    if focal_loss: 
        cost = (1 - torch.exp(-cost)) ** gamma * cost
        cost = torch.mean(cost[~torch.isnan(cost)])
    return cost

def test_data_loader(data_source, params):
    """ Create testing data loader.
            @param: data_source - Data dir or NAME:VERSION of registered dataset on AzureML
            @param: params - dict of parameters: imgWH, is_RGB, batch_size, num_workers
            @return: data loader (torch)
    """
    print(f"Getting data: {data_source}")

    p = Path(data_source)
    assert p.exists(), f"Error! Wrong data path: {p}"

    ds = dataset.lmdbDataset(
        root      = str(p), 
        transform = dataset.resizeNormalize(params['imgWH']), 
        rgb       = params['is_RGB'], 
        test_mode = True
    )
    assert ds, "Error! Failed getting data"

    shuffle, sampler = None, None
    if IS_HVD:
        sampler = torch.utils.data.DistributedSampler(dataset=ds, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        shuffle = True

    loader = torch.utils.data.DataLoader(
        ds, 
        shuffle     = shuffle,
        sampler     = sampler,
        batch_size  = params['batch_size'], 
        num_workers = params['num_workers'], 
        pin_memory  = True
    )
    assert loader, "Error! Can't get data"
    assert len(loader) > 0, "Error! Data len=0"
    return loader

def train_data_loader(data_source, params):
    """ Create training data loader.
            @param: data_source - Data dir or NAME:VERSION of registered dataset on AzureML
            @param: params - dict of parameters: imgWH, keep_ratio, is_RGB, batch_size, num_workers
            @return: data loader (torch)
    """
    print(f"Getting data: {data_source}")

    p = Path(data_source)
    assert p.exists(), f"Error! Wrong data path: {p}"

    ds = dataset.lmdbDataset(
        root    = str(p),
        rgb     = params['is_RGB']
    )
    assert ds, "Error! Failed getting data"

    imgW, imgH = params['imgWH']
    collate_fn = dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=params['keep_ratio'])

    shuffle, sampler = None, None
    if IS_HVD:
        sampler = torch.utils.data.DistributedSampler(dataset=ds, num_replicas=hvd.size(), rank=hvd.rank())
    elif not params['random_sample']:
        sampler = dataset.randomSequentialSampler(ds, params['batch_size'])
    else:
        shuffle = True

    loader = torch.utils.data.DataLoader(
        ds,
        shuffle     = shuffle,
        sampler     = sampler,
        batch_size  = params['batch_size'],
        num_workers = params['num_workers'],
        pin_memory  = True,
        collate_fn  = collate_fn
    )
    assert loader, "Error! Can't get data"
    assert len(loader) > 0, "Error! Data len=0"
    return loader

def optimizer_build(model, net_module, lr):
    """ Building training optimizer.
            @param: model  - Training model
            @param: net_module - Net module
            @param: lr     - Learning rate
            @return: optimizer (Adam/Adadelta/RMSprop)
    """
    weight_decay = net_module.weight_decay if hasattr(net_module, 'weight_decay') else 0
    if net_module.adam:
        return optim.Adam(model.parameters(), lr=lr, betas=(net_module.beta1, 0.999), weight_decay=weight_decay)
    elif net_module.adadelta:
        return optim.Adadelta(model.parameters(), weight_decay=weight_decay)
    return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

def set_optimizer(model, net_module, params, lr, fp16_allreduce=False, reset=False):
    """ Create or reset optimizer.
            @param: model   - training model
            @param: net_module  - Net module
            @param: params - dict of all external parameters
            @param: lr      - learning rate
            @param: fp16_allreduce - flag of horovod compression, if False - compression is none
            @param: reset   - flag to reset optimizer.
            @return: None (optimizer is added to dict: params)
    """
    if IS_HVD:
        if reset:
            for param_group in params['optimizer'].param_groups:
                param_group['lr'] = lr
        else:
            params['optimizer'] = optimizer_build(model, net_module, lr)
            # Horovod: (optional) compression algorithm.
            compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none
            params['optimizer'] = hvd.DistributedOptimizer(
                optimizer=params['optimizer'], 
                named_parameters=model.named_parameters(), 
                compression=compression
            )
        hvd.broadcast_optimizer_state(params['optimizer'], root_rank=0)
    else:
        params['optimizer'] = optimizer_build(model, net_module, lr)

def to_hvd(metrics: dict, names: list) -> None:
    """ Reduce metrics for horovod. Allreduce is an operation that aggregates data among multiple processes 
        and distributes results back to them. Allreduce is used to average dense tensors.
        Explanation: https://horovod.readthedocs.io/en/stable/concepts_include.html
            @param: metrics - dict of metrics
            @param: names   - metric names
            @return: None
    """
    if IS_HVD:
        for name in names:
            metrics[name] = hvd.allreduce(torch.tensor(metrics[name], dtype=torch.int64), name=name, op=hvd.Sum).item()

def validation(model, val_loader, params, msg='Evaluation', show_details=False):
    """ Validation model on testing dataset.
            @param: model   - training model
            @param: val_loader - loader of testing data
            @param: params - dict of all external parameters
            @param: msg   - Message in progress bar
            @param: show_details - Flag to show accuracy per US State, Block, Score
            @return: dict of metrics (accuracy, ...)
    """

    if not val_loader:
        return None

    print(f"Start validation: net type: {params['net_type']}, number of samples: {len(val_loader.dataset)}, Time: {str(datetime.now())}")

    flag = show_details and not IS_HVD

    names=['plate_total','plate_correct','plate_correct_0=O','char_total','char_correct']
    acc_names = ['char_acc','plate_acc','plate_acc_0=O']
    m_ = {n: 0 for n in names + acc_names}
    m = {
        'test_loss': 0.,
        'total':   m_,
        'nostack': m_.copy(),
        'stack':   m_.copy(),
        # Binary classification: stack/nostack
        'classify': {
            'n_correct': 0,
            'TP': 0,
            'FN': 0,
            'FP': 0,
            'Precision': 0,
            'Recall': 0,
            'F1': 0,
            'acc': 0
        },
    }
    m['total']['plate_total'] = len(val_loader.dataset)
    if flag:
        m['state'] = {}
        m['block'] = {}
        m['score'] = {}
    
    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    loss_avg_test  = Averager()
    loss_avg_s     = Averager()
    loss_avg_ocr   = Averager()
    loss_avg_class = Averager()

    max_iter = len(val_loader)
    print(f"max_iter = {max_iter}")
    time_count = 0

    errors = []

    for data in tqdm(val_loader, desc=msg):
        cpu_images = data[0]
        cpu_texts  = data[1]
        states     = data[2]
        hash_ids   = data[3]
        blocks     = data[4]
        scores     = data[5]

        batch_size = cpu_images.size(0)
        images = cpu_images.to(params['device'])
        
        cpu_texts_ori, cpu_texts = utils.text_transform(texts=cpu_texts, label=params['net_label'])
        cpu_texts_decode = [utils.one_line(txt).decode('utf-8', 'strict') for txt in cpu_texts_ori]

        if params['net_label'] == 'two-lines':
            text, length = params['converter'].encode([txt[0] for txt in cpu_texts])
            text_s, length_s = params['converter'].encode([txt[1] for txt in cpu_texts])

            t0 = time()
            preds, preds_s = model(images)
            time_count += time() - t0

            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
            cost = params['criterion'](preds, text, preds_size, length)
            cost_s = params['criterion'](preds_s, text_s, preds_size, length_s)
            loss_avg_test.add(cost)
            loss_avg_s.add(cost_s)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            _, preds_s = preds_s.max(2)
            preds_s = preds_s.transpose(1, 0).contiguous().view(-1)

            sim_pred = params['converter'].decode_2heads(preds.data, preds_s.data, preds_size.data, raw=False)
            raw_pred = params['converter'].decode_2heads(preds.data, preds_s.data, preds_size.data, raw=True)[:params['net_n_val_disp']]
        elif params['net_label'] == 'classify':
            text, length = params['converter'].encode(cpu_texts)
            class_targets = torch.FloatTensor([1 if '[' in txt else 0 for txt in cpu_texts_ori])
            classes = class_targets.to(params['device'])
            class_targets = np.array(class_targets)

            t0 = time()
            preds, class_preds = model(images)

            preds = preds.permute(2, 0, 1) # permute to [T, b, C] for CTCloss function
            time_count += time() - t0

            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
            cost = params['criterion'](preds, text, preds_size, length)

            cost = if_focal_loss(cost, params['focal_loss'], params['gamma'])
            cost_classify = params['criterion_1'](class_preds[0], classes)  * params['classify_lambda']

            loss_avg_ocr.add(cost)
            loss_avg_class.add(cost_classify)

            cost += cost_classify
            loss_avg_test.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            sim_pred = params['converter'].decode(preds.data, preds_size.data, raw=False,always_list = True)
            raw_pred = params['converter'].decode(preds.data, preds_size.data, raw=True,always_list = True)[:params['net_n_val_disp']]

            class_preds = np.array(class_preds.values.cpu().data.numpy() >= 0,dtype = int)
            m['classify']['n_correct'] +=  sum (class_preds == class_targets)
            m['classify']['TP'] += sum((class_preds == class_targets)*(class_targets == 1))
            m['classify']['FN'] += sum((class_preds != class_targets)*(class_targets == 1))
            m['classify']['FP'] += sum((class_preds != class_targets)*(class_targets == 0))
        else:
            text, length = params['converter'].encode(cpu_texts)
            t0 = time()
            # always return [b, c, T] from the network. This is crucial for DataParallel to work
            preds = model.forward(images)
            preds = preds.permute(2, 0, 1) # permute to [T, b, C] for CTCloss function
            time_count += time() - t0

            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
            cost = params['criterion'](preds, text, preds_size, length)
            loss_avg_test.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            sim_pred = params['converter'].decode(preds.data, preds_size.data, raw=False, always_list = True)
            raw_pred = params['converter'].decode(preds.data, preds_size.data, raw=True , always_list = True)[:params['net_n_val_disp']]
        
        for j in range(len(cpu_texts_ori)):
            ori = cpu_texts_ori[j]
            state = states[j]
            hash_id = hash_ids[j]
            block = blocks[j].item()
            score = round(scores[j].item(), 1)

            pred = sim_pred[j]
            target = cpu_texts_decode[j]
            r = SequenceMatcher(None, target, pred).ratio()

            key = 'stack' if '[' in ori else 'nostack'
            m[key]['plate_total'] += 1
            m[key]['char_correct'] += r * len(target)
            m[key]['char_total'] += len(target)
            
            kv = None
            if flag:
                kv = [('state', state), ('block', block), ('score', score)]
                for k, v in kv:
                    if v not in m[k]:
                        m[k][v] = {'total': 0, 'correct': 0}
                    m[k][v]['total'] += 1

            if pred == target:
                m[key]['plate_correct'] += 1
                if flag:
                    for k, v in kv:
                        m[k][v]['correct'] += 1
            elif params.get('error_folder', None): 
                error_save(errors, target, pred, cpu_images[j], hash_id, str(params['error_folder']))
            
            if pred.replace('O', '0') == target.replace('O', '0'):
                m[key]['plate_correct_0=O'] += 1

    # Calc. acc. metrics
    for key in ['stack','nostack']:
        to_hvd(m[key], names=names)
    
    def _calc_acc(m_):
        for k1, k2, k3 in [('plate_acc','plate_correct','plate_total'), ('plate_acc_0=O','plate_correct_0=O','plate_total'), ('char_acc','char_correct','char_total')]:
            m_[k1] = m_[k2]/m_[k3] if m_[k3] else 0
    
    _calc_acc(m['nostack'])
    if params['net_label'] != "no-stack":
        _calc_acc(m['stack'])
    
    for key in ['plate_correct','plate_correct_0=O','char_correct','char_total']:
        m['total'][key] = m['stack'][key] + m['nostack'][key]
    _calc_acc(m['total'])
    
    # Calc. Loss
    m['test_loss'] = ((loss_avg_test.val() * m['nostack']['plate_total'] + loss_avg_s.val() * m['stack']['plate_total'])/m['total']['plate_total']).item()

    # Calc. metrics of binary classification: stacked / no stacked
    if params['net_label'] == 'classify':
        m_ = m['classify']
        to_hvd(m_, names=['n_correct','TP','FN','FP'])
        m_['acc']       = m_['n_correct'] / m['total']['plate_total']
        m_['Recall']    = m_['TP'] / (m_['TP'] + m_['FN']) if m_['TP'] + m_['FN'] != 0 else 0
        m_['Precision'] = m_['TP'] / (m_['TP'] + m_['FP']) if m_['TP'] + m_['FP'] != 0 else 0
        m_['F1']        = 2 * m_['Precision'] * m_['Recall'] / (m_['Precision'] + m_['Recall']) if m_['Recall'] * m_['Precision'] != 0 else 0

    # Calc Acc. per US State, Block, Score
    others = None
    if flag:
        others = [ ('state', US_STATES), ('block', BLOCKS), ('score', SCORES) ]
        for k, vv in others:
            for v in vv:
                if v in m[k]:
                    h = m[k][v]
                    h['acc'] = h['correct']/h['total'] if h['total'] != 0 else 0
    
    # Print num samples: stack and nostack
    for k in ['nostack','stack']:
        print(f"Num samples {k:7s} : {m[k]['plate_total']:6d}/{m['total']['plate_total']:d}")

    # Print examples of orig. and detected texts
    for raw_pred, pred, gt, gt_ori in zip(raw_pred, sim_pred, cpu_texts_decode, cpu_texts_ori):
        print('%-20s => %-10s, gt: %-10s, gt origin: %-20s' % (raw_pred, pred, gt, gt_ori))
    
    # Print Acc. for nostack, stack, total
    def _print_acc(m_, title, acc_names, fps):
        print(f"{title:27s} " + ', '.join([f"{key}: {m_[key]:.4f}" for key in acc_names]) + f", speed: {fps:d}fps")
    fps = int(m['total']['plate_total'] / time_count)
    _print_acc(m['nostack'], 'No stack::', acc_names, fps)
    _print_acc(m['stack'], '   Stack::', acc_names, fps)
    _print_acc(m['total'], f"   Total::Val loss: {m['test_loss']:.4f},", acc_names, fps)

    # Print metrics of binary classification: stacked / no stacked
    if params['net_label'] == 'classify':
        m_ = m['classify']
        print('Classify::Acc %.4f, Recall %.4f, Prec %.4f, F1 %.4f. Loss OCR: %.6f. Loss Classify: %.6f. Loss Ratio: %.4f' % (
            m_['acc'], m_['Recall'], m_['Precision'], m_['F1'], loss_avg_ocr.val(), loss_avg_class.val(), loss_avg_class.val()/loss_avg_ocr.val()))

    # Print Acc. for US States, Blocks, Scores
    if flag:
        for k, vv in others:
            print(f"\n{k.title():25s}\t{'Count':6s}\t{'Acc.%'}\n")
            for v in vv:
                h = m[k].get(v, {'total': 0, 'correct': 0, 'acc': 0})
                acc = h['acc']
                print(f"{str(v):20s}\t{h['total']:6d}\t{100*acc:6.2f}%")
    
    return m

def train(model, train_iter, params):
    """ Training model on training dataset.
            @param: model   - training model
            @param: train_iter - iterator of training data loader
            @param: params - dict of all external parameters
            @return: cost value
    """
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data[0], data[1]
    batch_size = cpu_images.size(0)
    # utils.loadData(image, cpu_images)
    images = cpu_images.to(params['device'])

    params['optimizer'].zero_grad()
    
    cpu_texts_ori, cpu_texts = utils.text_transform(texts=cpu_texts, label=params['net_label'])

    if params['net_label'] == 'two-lines':
        text, length = params['converter'].encode([text[0] for text in cpu_texts])
        text_s, length_s = params['converter'].encode([text[1] for text in cpu_texts])

        preds, preds_s = model(images)

        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        preds_size_s = Variable(torch.LongTensor([preds_s.size(0)] * batch_size))

        cost = params['criterion'](preds, text, preds_size, length) * (1 - params['stack_weight'])
        cost += params['criterion'](preds_s, text_s, preds_size_s, length_sreduction = params['reduction_loss'])  * params['stack_weight']
        cost = if_focal_loss(cost, params['focal_loss'], params['gamma'])
    elif params['net_label'] == 'classify':
        text, length = params['converter'].encode(cpu_texts)
        class_targets = torch.FloatTensor([1 if '[' in text else 0 for text in cpu_texts_ori])
        classes = class_targets.to(params['device'])

        preds, class_preds = model(images)

        # print(f'pre pred size = {preds.shape}')
        preds = preds.permute(2, 0, 1) # permute to [T, B, C]
        # print(f'post pred size = {preds.shape}')
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

        cost  = params['criterion'](preds, text, preds_size, length)

        cost = if_focal_loss(cost, params['focal_loss'], params['gamma'])
        cost_classify = params['criterion_1'](class_preds.values, classes) * params['classify_lambda']
        cost += cost_classify
    else:
        text, length = params['converter'].encode(cpu_texts)

        # prunner = FilterPrunner(model)
        # preds = prunner.forward(images)
        # prunner.normalize_ranks_per_layer()
        preds = model.forward(images)
        preds = preds.permute(2, 0, 1)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = params['criterion'](preds, text, preds_size, length)
        cost = if_focal_loss(cost, params['focal_loss'], params['gamma']) 

    assert(~torch.isnan(cost))
    cost.backward()
    params['optimizer'].step()

    return cost

def save_model_weights(model, model_path):
    """ Saving model weights to file. 
        If training process is using horovod then save weight on master node only.
            @param: model   - Model
            @param: model_path - Outtput model path
            @return: None
    """
    if not IS_HVD or hvd.rank()==0: # do it on master node only
        torch.save(model.state_dict(), str(model_path))
        print(f"Weights saved! {str(model_path)}")

def get_item(val):
    if isinstance(val, (Variable, torch.Tensor)):
        return val.item()
    return val

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

def train_logging(log_file, record=None):
    """ Logging training metrics as: 1) AzureML metrics 2) .csv file in output folder 3) TensorBoard SummaryWriter
            @param: log_file - Path to log file
            @param: record  - list of training metrics; if None then only header will be saved to .csv file
            @return: None
    """
    if not record:
        write_log(file=log_file, row=['epoch','epochs','iter','iters','train_loss','batch','time'], mode='w')
    else:
        record = [get_item(x) for x in record]
        # Rounding train_loss for easier view in logs
        train_loss = record[4] = round(record[4], 4)
        # Write logs
        write_log(file=log_file, row=record, mode='a')
        if ONLINE:
            aml.log_metric('train_loss', train_loss, f"Training loss")
        else:
            writer.add_scalar(f"train_loss", train_loss, record[0])
        print("[{0:d}/{1:d}][{2:d}/{3:d}] Loss: {4:f} Batch: {5:d} Time: {6:s}".format(*record))

def test_logging(log_file, epoch=None, batch=None, metrics=None):
    """ Logging testing metrics as: 1) AzureML metrics 2) .csv file in output folder 3) TensorBoard SummaryWriter
            @param: log_file - Path to log file
            @param: epoch   - Epoch number
            @param: batch   - Batch number
            @param: metrics - Dict. of testing metrics. If None then only header will be saved to .csv file
            @return: None
    """
    nn0 = ['test_loss','model_saved','learning_rate']
    nn1 = ['total','nostack','stack']
    nn2 = ['plate_acc','plate_acc_0=O','char_acc']
    if not metrics:
        write_log(file=log_file, row = ['epoch','batch'] + nn0 + [f"{n1}_{n2}" for n1 in nn1 for n2 in nn2], mode='w')
    else:        
        # Rounding metrics for easier view in logs
        metrics['test_loss'] = round(get_item(metrics['test_loss']), 4)
        for n1 in nn1:
            for n2 in nn2:
                metrics[n1][n2] = round(get_item(metrics[n1][n2]), 4)

        # Write logs
        write_log(file=log_file, row = [epoch, batch] + [metrics[n] for n in nn0] + [metrics[n1][n2] for n1 in nn1 for n2 in nn2], mode='a')
        if ONLINE:
            for n in nn0:
                aml.log_metric(n, metrics[n])
            for n1 in nn1:
                for n2 in nn2:
                    aml.log_metric(f"{n1}_{n2}", metrics[n1][n2])
        else:
            writer.add_scalar(f"accuracy", metrics['total']['plate_acc'], epoch)
            writer.add_scalar(f"val_loss", metrics['test_loss'], epoch)

def register_model(model_path, net_type, scheduler):
    if ONLINE and scheduler.best_accuracy > 0.5 and (not IS_HVD or hvd.rank()==0): # do it on master node only:
        # Register a new model on AzureML if no model with better accuracy and smaller version on the same train/valid datasets.
        aml.register_model(
            weights_path = str(model_path),
            model_name   = f"{net_type}_ALPR-OCR",
            descr        = f"ALPR OCR {net_type} Model",
            tags = {
                aml.ACC_TAG: scheduler.best_accuracy_str
            }
        )

class Averager(object):
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

class LearningRateScheduler():
    """ Learning rate scheduler.
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