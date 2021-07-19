import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # Add this line to avoid warning about flatting the LSTM layer
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class BidirectionalGRU(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, num_layers=1):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, num_layers = num_layers, dropout=0.1, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class Embedding(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        T, b, h = input.size()
        t_rec = input.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def checksize(net, x):
    net.eval()
    if isinstance(net(x), tuple):
        final_size = [list(out.size()) if hasattr(out,'size') else out[0].size() for out in net(x)]
    else:
        final_size = list(net(x).size())
    print ('CNN sizes:')
    print('%15s: %s'%('Input',str(list(x.size()))))
    from os.path import splitext
    for layer in net.cnn:
        x = layer(x)
        _, layer_name = splitext(str(type(layer))[:-2])
        print('%15s: %s'%(layer_name[1:],str(list(x.size()))))
    print ('Final Output Size:', final_size)
    return final_size[0] # number of sequence steps

def print_model_info(model, params, is_layers=False):
    def print_layers(model):
        print("Layers:")
        print(model)

    def print_count_parameters(title, model):
        count_params = sum(p.numel() for p in model.parameters())#if p.requires_grad
        print(f"{title} Parameters: {count_params}")

    def print_number_of_steps(model):
        x = torch.randn(1, params.nc, params.imgH, params.imgW)
        steps = checksize(model, x)
        print(f"Number of Steps: {steps}")

    def print_weights_shape(model):
        print("Weights Shape:")
        layers = model.state_dict()
        for layer in layers:
            print(layer + ':'+ str(list(layers[layer].shape)))
    
    if is_layers:
        print_layers(model)
        print_weights_shape(model)
    
    print_count_parameters("Total", model)

    if hasattr(model, 'cnn'):
        print_count_parameters("CNN", model.cnn)
        print_number_of_steps(model)
    
    if hasattr(model, 'rnn'):
        print_count_parameters("RNN", model.rnn)