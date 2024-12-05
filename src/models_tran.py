
import copy
import torch
import torch.nn as nn
from torch.nn import ModuleList, Linear, Dropout, LayerNorm, MultiheadAttention

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, ff = True,
                device=None, dtype=None):
        super(TransformerEncoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = MultiheadAttention(embed_dim = d_model, num_heads = nhead, dropout=dropout,
                                            batch_first=True, **factory_kwargs)
        layer_norm_eps = 1e-5

        self.ff = ff

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)

        if ff:
            self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.dropout2 = Dropout(dropout)
            self.activation = nn.GELU()

    def forward(self, src, src_mask = None, src_key_padding_mask = None, need_weights=False):
        x = src
        if need_weights:
            attn_out, weight = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                            need_weights=True, average_attn_weights=False)
        else:
            attn_out = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                        need_weights=False)[0]
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        if self.ff:
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            ff_out = self.dropout2(ff_out)
            x = self.norm2(x + ff_out)
        
        if need_weights:
            return x, weight
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, self_attn_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(self_attn_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask = None, src_key_padding_mask = None, need_weights=False):
        x = src
        weights = []
        for i in range(self.num_layers):
            if need_weights:
                x, weight = self.layers[i](x, mask, src_key_padding_mask, need_weights=True)
                weights.append(weight)
            else:
                x = self.layers[i](x, mask, src_key_padding_mask, need_weights=False)
        if need_weights:
            return x, weights
        return x
        

def test():
    d_model = 512
    n_head = 8
    dim_feedforward = 2048
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self_attn_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
    transformer = TransformerEncoder(self_attn_layer, 6)
    transformer.to(device)
    x = torch.randn(10, 32, d_model).to(device)
    out, weights = transformer(x, need_weights=True)
    print(out.shape)
    for weight in weights:
        print(weight.shape)


if __name__ == "__main__":
    test()