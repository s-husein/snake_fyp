import torch.nn as nn
from torchvision.datasets import ImageFolder



def make_cnn(dataset, hid_layers=[64, 64], act_fn='relu', max_pool=None, avg_pool=None, pooling_after_layers=2,
             dropout=0.2, batch_norm=True, groups=1, bias=False, conv_layers=[[32, 3, 1]], return_features=False):

    img = dataset.__getrawimage__()
    input_shape = img.shape

    if len(input_shape) < 3:
        input_shape = (1, *input_shape)


    in_chann, inp_h, inp_w = input_shape
    activation_fun = {'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}
    layers = []

    for ind, conv in enumerate(conv_layers):
        out_chann, filter_size, stride = conv
        layers.append(nn.Conv2d(in_chann, out_chann, filter_size, stride, groups=groups, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_chann))
        layers.append(activation_fun[act_fn])

        # Update dimensions
        inp_h = (inp_h - filter_size) // stride + 1
        inp_w = (inp_w - filter_size) // stride + 1

        if max_pool is not None and ((ind + 1) % pooling_after_layers == 0 or ind == (len(conv_layers) - 1)):
            layers.append(nn.MaxPool2d(max_pool[0], max_pool[1]))
            inp_h = (inp_h - max_pool[0]) // max_pool[1] + 1
            inp_w = (inp_w - max_pool[1]) // max_pool[1] + 1

        if avg_pool is not None:
            layers.append(nn.AdaptiveAvgPool2d(avg_pool))
            inp_h, inp_w = avg_pool
            print(f'Adaptive AvgPool: h = {inp_h}, w = {inp_w}, c = {out_chann}')

        in_chann = out_chann

    layers.append(nn.Flatten())
    feat_dim = inp_h * inp_w * in_chann

    if return_features:
        return nn.Sequential(*layers), feat_dim

    layers.append(nn.Linear(feat_dim, hid_layers[0]))
    layers.append(activation_fun[act_fn])

    for in_dim, out_dim in zip(hid_layers[:-1], hid_layers[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers.append(activation_fun[act_fn])

    num_of_classes = len(dataset.classes)
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


class CNNLSTM(nn.Module):
    def __init__(self, cnn_backbone, feature_dim, lstm_hidden=128, lstm_layers=1,
                 hid_layers=[128], num_classes=5, dropout=0.3, act_fn='relu'):
        super().__init__()
        self.cnn = cnn_backbone
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)

        activation_fun = {
            'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(), 'tanh': nn.Tanh(), 'elu': nn.ELU()
        }

        classifier_layers = []
        prev_dim = lstm_hidden
        for h_dim in hid_layers:
            classifier_layers.append(nn.Linear(prev_dim, h_dim))
            classifier_layers.append(activation_fun[act_fn])
            classifier_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)  # [B*T, F]
        features = features.view(B, T, -1)  # [B, T, F]
        out, (hn, cn) = self.lstm(features)
        return self.classifier(out[:, -1])


def make_cnn_lstm(dataset, lstm_hidden=128, lstm_layers=1, dropout=0.3, act_fn='relu', **cnn_kwargs):
    cnn, feat_dim = make_cnn(dataset, return_features=True, **cnn_kwargs)
    num_classes = len(dataset.classes)
    return CNNLSTM(
        cnn_backbone=cnn,
        feature_dim=feat_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        hid_layers=cnn_kwargs.get('hid_layers', [128]),  # Reuse same hid_layers from params
        num_classes=num_classes,
        dropout=dropout,
        act_fn=act_fn
    )
        