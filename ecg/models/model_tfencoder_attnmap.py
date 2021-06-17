# https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/
# conv1dを並列に並べる + tfencoder + lstm

import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights

class Model_TFEncoder_AttnMap(nn.Module):
    def __init__(self, in_channel=12, ninp=512, nhead=4, nhid=1024, dropout=0.1, nlayers=2):
        super(Model_TFEncoder_AttnMap, self).__init__()

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer = TransformerEncoder(encoder_layers, nlayers)

        d_model = ninp
        self.top_layer1 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=3, padding=1),
                             nn.ReLU(inplace=True),
                         )

        self.top_layer2 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=5, padding=2),
                             nn.ReLU(inplace=True),
                         )

        self.top_layer3 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=7, padding=3),
                             nn.ReLU(inplace=True),
                         )

        self.top_layer4 = nn.Sequential(
                             nn.Conv1d(in_channel, d_model//4, kernel_size=11, padding=5),
                             nn.ReLU(inplace=True),
                         )

        self.top_conv = nn.Sequential(
                             nn.Conv1d(d_model, d_model, kernel_size=1),
                             nn.ReLU(inplace=True),
                         )
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=1,
                                    batch_first=True, bidirectional=True)


        self.bottom_linear = nn.Sequential(
                                 nn.Linear(d_model, d_model//2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model//2, 1)
                             )


    def forward(self, x):
        x = x.squeeze(1)
        x1 = self.top_layer1(x)
        x2 = self.top_layer2(x)
        x3 = self.top_layer3(x)
        x4 = self.top_layer4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.top_conv(x).permute(2, 0, 1)
        x, _ = self.lstm(x)
        x_t, w = self.transformer(x)

        x = x_t.permute(1, 2, 0)
        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[0], -1)
        x = self.bottom_linear(x)
        return x, w