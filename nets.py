import torch.nn as nn
from torchvision.datasets import ImageFolder



def make_cnn(dataset, hid_layers = [64, 64],
            act_fn='relu', max_pool = None, avg_pool = None, pooling_after_layers = 2, dropout = 0.2, batch_norm=True,
            groups =1, bias=False, conv_layers=[[32, 3, 1],
                                                [16, 3, 1]]):
    
    img = dataset.__getitem__(0)[0]
    input_shape = img.shape

    num_of_classes = len(dataset.classes)

    layers = []
    activation_fun = {'relu': nn.ReLU(inplace=True), 'softplus':nn.Softplus(), 'tanh':nn.Tanh(), 'elu': nn.ELU()}

    assert pooling_after_layers <= len(conv_layers), 'Max pooling cannot be added after last convolution layer..'

    in_chann, inp_h, inp_w = input_shape
    for ind, conv in enumerate(conv_layers):
        out_chann, filter_size, stride = conv
        layers.append(nn.Conv2d(in_chann, out_chann, filter_size, stride, groups=groups, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_chann))
        layers.append(activation_fun[act_fn])

        out_h = (inp_h - filter_size)//stride + 1
        out_w = (inp_w - filter_size)//stride + 1
        inp_h = out_h
        inp_w = out_w

        if max_pool is not None and ((ind+1) % pooling_after_layers == 0 or ind == (len(conv_layers) - 1)):
            layers.append(nn.MaxPool2d(max_pool[0], max_pool[1]))
            out_h = (inp_h - max_pool[0])//max_pool[1] + 1
            out_w = (inp_w - max_pool[0])//max_pool[1] + 1
            inp_h = out_h
            inp_w = out_w
        if avg_pool is not None:
            layers.append(nn.AvgPool2d(avg_pool[0], avg_pool[1]))
            out_h = (inp_h - avg_pool[0])//avg_pool[1] + 1
            out_w = (inp_w - avg_pool[0])//avg_pool[1] + 1
            inp_h = out_h
            inp_w = out_w
            print(f'h = {inp_h}, w = {inp_w}, c = {in_chann}')
        in_chann = out_chann

    layers.append(nn.Flatten())
    layers.append(nn.Linear(inp_h*inp_w*in_chann, hid_layers[0]))
    layers.append(activation_fun[act_fn])
    if len(hid_layers) > 1:
        dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
        for in_dim, out_dim in list(dim_pairs):
            layers.append(nn.Linear(in_dim, out_dim))
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
            layers.append(activation_fun[act_fn])

    layers.append(nn.Linear(hid_layers[-1], num_of_classes))
    return nn.Sequential(*layers)


def make_dnn(dataset: ImageFolder, model, hid_layers = [64, 64],
            act_fn='relu'):
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh(), 'elu': nn.ELU()}
    num_of_classes = len(dataset.classes)

    if len(hid_layers) > 1:
        dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
        for in_dim, out_dim in list(dim_pairs):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_fun[act_fn])

    layers.append(nn.Linear(hid_layers[-1], num_of_classes))
    return nn.Sequential(*layers)

        