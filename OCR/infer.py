import torch
from torch.autograd import Variable
import tool.utils as utils
import dataset
from PIL import Image

from layers import checksize
import argparse
import os
from .tool import BeamSearch

if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required = True, help = 'model path')
    parser.add_argument('--img_path', type = str, required = True, help = 'image path')
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.img_path

    args.cfg = os.path.dirname(args.model_path)+'.py'
    name = os.path.basename(args.model_path)
    args.net_type, name  = name[:name.index('_')], name[name.index('_')+1:]

    # import network architecture and hyper-parameters
    params = True
    exec("import %s as params" % args.cfg[:-3].replace('/','.'))

    if params.cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # net init
    nclass = len(params.alphabet) + 1

    model = True
    exec("import %s as net" % args.cfg[:-3].replace('/','.'))

    def net_init(net_type):
        nclass = len(params.alphabet) + 1

        model = model.OCR(params.imgH, params.nc, nclass, params.nh)
        if args.model_path != '':
            print('loading pretrained model from %s' % args.model_path)
            if params.multi_gpu:
                model = torch.nn.DataParal
            model.load_state_dict(torch.load(args.model_path,map_location=torch.device(device)))

        return model


    model = net_init(args.net_type)
    #print("Layers:")
    #print(model)
    #
    #model.eval()
    #print("Weights Shape:")
    #layers = model.state_dict()
    #for layer in layers:
    #    print(layer+ ':'+ str(list(layers[layer].shape)))

    x = torch.randn(1, params.nc, params.imgH, params.imgW)
    if hasattr(model, 'cnn'): # has convolutional layers
        steps = checksize(model,x)
        print("Number of Steps:", steps)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:",total_params)
    if hasattr(model, 'cnn'):
        cnn_params = sum(p.numel() for p in model.cnn.parameters()) #if p.requires_grad
        print("CNN Parameters:",cnn_params)

    if hasattr(model, 'rnn'):
        rnn_params = sum(p.numel() for p in model.rnn.parameters()) #if p.requires_grad
        print("RNN Parameters:",rnn_params)

    converter = utils.strLabelConverter(params.alphabet)

    transformer = dataset.resizeNormalize((params.imgW, params.imgH))
    if params.nc == 1:
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path)

    print(image.size)
    image = transformer(image)
    print(image.size)

    if params.cuda and torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    #print(next(model.parameters()).is_cuda) #check whether model is on cuda
    model.eval()
    preds = model(image)

    preds = preds.squeeze(1)#.transpose(1, 0)
    preds = preds.cpu().detach().numpy()
    res = BeamSearch.ctc_beamsearch(preds, '-'+params.alphabet, 10)
    print(res)

    #_, preds = preds.max(2)
    #preds = preds.transpose(1, 0).contiguous().view(-1)
    #
    #preds_size = Variable(torch.LongTensor([preds.size(0)]))
    #raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    #sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('%-20s => %-20s' % (raw_pred, sim_pred))
