import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)


def buffered_future_mask(curr_len, max_len, device):
    future_mask = torch.triu(
        fill_with_neg_inf(torch.zeros([max_len, max_len])), 1
    )
    future_mask = future_mask.unsqueeze(0)

    future_mask = future_mask.to(device)
    return future_mask[:, :curr_len, :curr_len]


def drop_attn_mask(max_len, drop_layer, w):
    """Implement DropAttention(c) described in https://arxiv.org/abs/1907.11065
    """
    mask = drop_layer(torch.ones([max_len, max_len])).T
    idx = torch.arange(0, max_len)

    zeros_idx = idx[mask[0, :] == 0]

    for i in zeros_idx:
        for q in range(w):
            if i + q < max_len:
                for j in range(max_len):
                    mask[j, i + q] = 0.0

    mask = torch.where(mask == 0, 0, 1)
    return mask.unsqueeze(0)


class ResBlock(nn.Module):

    def __init__(self, obs_dim, fw_module):
        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)
        self.fw_module = fw_module

    def forward(self, x, **kwargs):
        return self.fw_module(self.norm(x), **kwargs)


class ResNet(nn.Module):

    def __init__(self, res_block):
        super().__init__()
        self.res_block = res_block

    def forward(self, x, **kwargs):
        return self.res_block(x, **kwargs) + x


class MLP(nn.Module):

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):

    def __init__(self, obs_dim, heads, p=0.0, w=2):
        super().__init__()
        self.heads = heads
        self.q_net = nn.Linear(obs_dim, obs_dim, bias=False)
        self.k_net = nn.Linear(obs_dim, obs_dim, bias=False)
        self.v_net = nn.Linear(obs_dim, obs_dim, bias=False)
        self.out_net = nn.Linear(obs_dim, obs_dim)
        self.scale_factor = obs_dim ** (-0.5)

        self.p = p
        self.w = w
        self.drop_attn = nn.Dropout1d(p / w)

    def forward(self, x, e=None, mask=None):
        b, n, _ = x.shape

        q = self.q_net(x)

        if e is not None:
            k = self.k_net(e)
            v = self.v_net(e)

        else:
            k = self.k_net(x)
            v = self.v_net(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale_factor

        if mask is not None:
            mask = mask.unsqueeze(0)
            scores = scores + mask

        att_scores = F.softmax(scores, dim=-1)

        if self.p > 0:
            mask = drop_attn_mask(max_len=n, drop_layer=self.drop_attn, w=self.w)
            mask = mask.unsqueeze(0).to(x)
            att_scores = torch.mul(att_scores, mask)

            att_scores = torch.div(att_scores, att_scores.sum(dim=-1, keepdim=True) + 1e-5)

        self_att = torch.einsum('bhij,bhjd->bhid', att_scores, v)
        return self.out_net(rearrange(self_att, 'b h n d -> b n (h d)'))


class EncLayers(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers, p, w=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResNet(ResBlock(obs_dim, Attention(obs_dim, heads, p, w))),
                        ResNet(ResBlock(obs_dim, MLP(obs_dim, hidden_dim)))
                    ]
                )
            )

    def forward(self, x, mask=None):
        for att_block, mlp_block in self.layers:
            x = att_block(x, mask=mask)
            x = mlp_block(x)
        return x


class DecLayers(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResNet(ResBlock(obs_dim, Attention(obs_dim, heads))),
                        ResNet(ResBlock(obs_dim, Attention(obs_dim, heads))),
                        ResNet(ResBlock(obs_dim, MLP(obs_dim, hidden_dim)))
                    ]
                )
            )

    def forward(self, x, e, mask=None):
        for att, enc_dec_att, mlp in self.layers:
            x = att(x, mask=mask)
            x = enc_dec_att(x, e=e)
            x = mlp(x)
        return x
