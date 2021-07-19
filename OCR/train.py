#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from time import time

import ocr_lib

if __name__ == "__main__":
    parser = ArgumentParser(description='OCR-SOR training/validation arguments.')
    parser.add_argument('--tag',              type=str,   default='ocr_sor', help="Tag for logging and model file suffix.")
    parser.add_argument('--cfg',              type=str,   required=True,     help="Net module name in ./models/ or path to .py module.")
    parser.add_argument('--hvd',              type=str,   default='True',    help="Whether to execute on azureml with horovod. Values: 1,T,True,true,0,F,False,false")
    parser.add_argument('--seed',             type=str,   default='42',      help='Random seed number')
    parser.add_argument('--model_path',       type=str,   default='',        help="Path to pre-trained model for transfer (incremental) learning")
    parser.add_argument('--trainroot',        type=str,   default='',        help="Path to training dataset")
    parser.add_argument('--valroot',          type=str,   default='',        help="Path to validation dataset")
    parser.add_argument('--batch_size',       type=str,   default='1024',    help="Batch size")
    parser.add_argument('--epochs',           type=str,   default='1000',    help="Number of epochs")
    parser.add_argument('--lr',               type=str,   default='1e-3',    help="Learning rate")
    parser.add_argument('--lr_period',        type=str,   default='10',      help="Is a period (num iterations) of --lr reduction if accuracy is not increased")
    parser.add_argument('--min_lr',           type=str,   default='1e-7',    help="Min value of learning rate to stop training.")
    parser.add_argument('--device',           type=str,   default='gpu',     help='Cuda device: CPU or GPU (max available nodes) or 0 or 1 or 0,1 or 0,1,2,3 or ... (choose nodes)')
    parser.add_argument('--num_workers',      type=str,   default='8',       help="Number of data loader workers")
    parser.add_argument('--display_interval', type=str,   default='100',     help='How many batches to wait before logging training status.')
    parser.add_argument('--val_interval',     type=str,   default='1000',    help='How many batches to wait before validation and logging metric.')
    parser.add_argument('--focal_loss',       type=str,   default='False',   help="If using focal loss. Values: 1,T,True,true,0,F,False,false")
    parser.add_argument('--gamma',            type=str,   default= '0.5',    help="Gamma of focal loss")
    parser.add_argument('--random_sample',    type=str,   default='True',    help="Whether to random sample the data. Values: 1,T,True,true,0,F,False,false")
    parser.add_argument('--register_model',   type=str,   default='False',   help="Whether to register model on azureml. Values: 1,T,True,true,0,F,False,false")
    parser.add_argument('--outdir',           type=str,   default='outputs', help="Output directoty with: model weights & logs.")
    args = parser.parse_args()

    print(f"Start model training. Time: {str(datetime.now())}")
    start_time = time()
    ocr_lib.print_args(args)
    now_date = str(datetime.now().date())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Init horovod
    ocr_lib.activate_hvd(ocr_lib.to_bool(args.hvd))

    device = ocr_lib.select_device(args.device, batch_size=int(args.batch_size))
    ocr_lib.set_random_seed(int(args.seed), device)

    net_module, net_type = ocr_lib.import_net_module(args.cfg)

    model = ocr_lib.init_model(net_module)

    out_model_stem = f"{net_type}_{args.tag}_{now_date}"

    params = {
        'now_date':         now_date,        
        'net_type':         net_type,
        'imgWH':            (net_module.imgW, net_module.imgH),
        'keep_ratio':       net_module.keep_ratio,
        'is_RGB':           (net_module.nc==3),
        'net_label':        net_module.label,
        'net_n_val_disp':   net_module.n_val_disp,
        'classify_lambda':  net_module.classify_lambda,
        'stack_weight':     (net_module.stack_weight if hasattr(net_module, 'stack_weight') else None),
        'device':           device,
        'batch_size':       int(args.batch_size),
        'num_workers':      int(args.num_workers),
        'focal_loss':       ocr_lib.to_bool(args.focal_loss),
        'gamma':            float(args.gamma),
        'converter':        ocr_lib.get_converter(net_module, device),
        'criterion':        ocr_lib.criterion_init(net_module, args.focal_loss, device),
        'criterion_1':      ocr_lib.criterion1_init(),
        'tag':              args.tag,
        'model_path':       args.model_path,
        'register_model':   ocr_lib.to_bool(args.register_model),
        'random_sample':    ocr_lib.to_bool(args.random_sample),
        'seed':             int(args.seed),
        'reduction_loss':   ('none' if ocr_lib.to_bool(args.focal_loss) else 'mean'),
        'lr':               float(args.lr),
        'lr_period':        int(args.lr_period),
        'min_lr':           float(args.min_lr),
        'optimizer':        None,
        'epochs':           int(args.epochs),
        'display_interval': int(args.display_interval),
        'val_interval':     int(args.val_interval),
        'out_model_path':   outdir.joinpath(f"{out_model_stem}.pth"),
        'train_csv_log':    outdir.joinpath(f"train_loss_{out_model_stem}.csv"),
        'test_csv_log':     outdir.joinpath(f"test_metrics_{out_model_stem}.csv")
    }

    ocr_lib.load_model_weights(model, params['model_path'], device)
    model = ocr_lib.model_prepare(model, net_module, device)

    train_loader = ocr_lib.train_data_loader(args.trainroot, params)
    val_loader = None
    if args.valroot:
        val_loader = ocr_lib.test_data_loader(args.valroot, params)
    
    scheduler = ocr_lib.LearningRateScheduler(
        lr = params['lr'],
        lr_period = params['lr_period'],
        min_lr = params['min_lr'],
        epochs = params['epochs']
    )

    batch = 0
    iters = len(train_loader)
    
    train_loss_avg = ocr_lib.Averager()

    # Write .csv log headers:
    ocr_lib.train_logging(params['train_csv_log'])
    if val_loader:
        ocr_lib.test_logging(params['test_csv_log'])

    # Show train loss after each display_interval iterations or at the end of each epoch.
    params['display_interval'] = min(params['display_interval'], iters)
    # Do model validation after each val_interval iterations or at the end of each epoch.
    params['val_interval'] = min(params['val_interval'], iters)
    
    ocr_lib.set_optimizer(model, net_module, params, lr=scheduler.lr)

    if val_loader and params['model_path']:
        metrics = ocr_lib.validation(model, val_loader, params, msg='Evaluation pre-trained model')
        
        scheduler.update(accuracy=metrics['total']['plate_acc'])
        ocr_lib.set_optimizer(model, net_module, params, lr=scheduler.lr, reset=True)
        print(f"\nContinue Training (transfer learning) with pre-trained model: {str(params['model_path'])}")
    else:
        print("\nStart Training (from scratch)")
    
    print(f"Learning rate: {scheduler.lr:.8f}")

    epoch = 0
    while epoch < params['epochs'] and not scheduler.stop_training:
        train_iter = iter(train_loader)
        iter_ = 0
        while iter_ < iters and not scheduler.stop_training:

            cost = ocr_lib.train(model, train_iter, params)
            train_loss_avg.add(cost)
            iter_ += 1
            batch += 1

            if batch % params['display_interval']==0:
                ocr_lib.train_logging(params['train_csv_log'], [epoch, params['epochs'], iter_, iters, train_loss_avg.val(), batch, str(datetime.now())])
                train_loss_avg.reset()
            
            if batch % params['val_interval']==0:
                if val_loader:
                    metrics = ocr_lib.validation(model, val_loader, params, msg=f'Evaluation at epoch: {epoch}')
                    metrics['learning_rate'] = scheduler.lr
                    metrics['model_saved'] = 0
                    
                    got_better_model = scheduler.update(accuracy=metrics['total']['plate_acc'], epoch=epoch, batch=batch)
                    if got_better_model:
                        ocr_lib.save_model_weights(model, params['out_model_path'])
                        metrics['model_saved'] = 1
                    elif not scheduler.stop_training:
                        ocr_lib.set_optimizer(model, net_module, params, lr=scheduler.lr, reset=True)

                    ocr_lib.test_logging(params['test_csv_log'], epoch, batch, metrics)                        
                else:
                    ocr_lib.save_model_weights(model, params['out_model_path'])
        epoch += 1
        
    if params['register_model']:
        ocr_lib.register_model(params['out_model_path'], net_type, scheduler)

    print(f"Elapsed time: {ocr_lib.format_time(time() - start_time)}")
    print(f"Done. Time: {str(datetime.now())}")
