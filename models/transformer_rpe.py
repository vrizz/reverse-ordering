import math

from models.transformer_utils import *


class PosEnc(nn.Module):

    def __init__(self, d_model, max_seq_len=20, p_drop=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p_drop)

        pos_enc = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

        # plt.imshow(pos_enc[:, :].numpy(), cmap='hot', interpolation='nearest')
        # plt.show()
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x, flip=True):
        x = x * math.sqrt(self.d_model)
        n = x.size(1)
        if flip is True:
            reverse_pos_enc = torch.flip(self.pos_enc[:, :n], [1])
            x = x + reverse_pos_enc
        else:
            x = x + self.pos_enc[:, :n]
        return x


class Encoder(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers, p=0.0, w=2):
        super().__init__()

        self.to_embed = nn.Linear(obs_dim, obs_dim)
        self.pos_enc = PosEnc(obs_dim)
        self.enc_layers = EncLayers(obs_dim, heads, hidden_dim, n_layers, p, w)

    def forward(self, x, mask=None):
        x = self.to_embed(x)
        x = self.pos_enc(x)  # reverse positional encoding is applied
        x = self.enc_layers(x, mask=mask)
        return x


class Decoder(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers):
        super().__init__()

        self.to_embed = nn.Linear(obs_dim, obs_dim)
        self.pos_enc = PosEnc(obs_dim)
        self.dec_layers = DecLayers(obs_dim, heads, hidden_dim, n_layers)
        self.to_out = nn.Linear(obs_dim, obs_dim)

    def forward(self, x, e, mask):
        x = self.to_embed(x)
        x = self.pos_enc(x, flip=False)  # standard positional encoding is applied
        x = self.dec_layers(x, e, mask)
        x = self.to_out(x)
        return x


class TransformerRPE(nn.Module):

    def __init__(self, obs_dim, heads, hidden_dim, n_layers_enc, n_layers_dec, p_enc=0.0, w=2):
        super().__init__()
        self.enc = Encoder(obs_dim, heads, hidden_dim, n_layers_enc, p=p_enc, w=w)
        self.dec = Decoder(obs_dim, heads, hidden_dim, n_layers_dec)

    def forward(self, src, trg, trg_mask, src_mask):
        e = self.enc(src, src_mask)
        return self.dec(trg, e, trg_mask)
