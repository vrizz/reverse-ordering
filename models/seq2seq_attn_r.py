import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()

        self.hidden_dim = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).to(x.device)

        out, hidden = self.gru(x, hidden)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class DecoderAttention(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, max_length=16):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.input_size)

        self.n_layers = n_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """

        :param input: shape [batch_size, input_size]
        :param hidden: shape [n_layers, batch_size, hidden_size]
        :param encoder_outputs: shape [batch_size, len_input_enc, hidden_size]
        :return: output: shape [batch_size, 1, output_size]
            h_n: shape [n_layers, batch_size, hidden_size]
        """
        cat_input_hidden = self.attn(torch.cat((input, hidden[-1]), dim=-1))
        att_scores = F.softmax(cat_input_hidden, dim=-1)  # shape [batch_size, max_len]

        if att_scores.shape[1] > encoder_outputs.shape[1]:
            att_scores = att_scores[:, :encoder_outputs.shape[1]]
            att_scores = torch.div(att_scores, att_scores.sum(dim=-1, keepdim=True) + 1e-5)

        reversed_enc_out = torch.flip(encoder_outputs, [1])
        self_att = torch.einsum('bi, bij-> bj', att_scores, reversed_enc_out)

        cat_input_att = torch.cat((input, self_att), dim=-1)
        output = F.relu(self.attn_combine(cat_input_att))

        output, h_n = self.gru(torch.unsqueeze(output, dim=1), hidden)

        output = self.out(output)

        return output, h_n


class Seq2SeqAttnR(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, max_length):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size, n_layers)
        self.decoder = DecoderAttention(input_size, hidden_size, output_size, n_layers, max_length)

        self.output_size = output_size

    def forward(self, enc_input, dec_input, target_len=4):
        batch_size = enc_input.size(0)

        enc_outputs, h_n = self.encoder(enc_input)
        dec_outputs = torch.zeros(batch_size, target_len, self.output_size).to(enc_outputs.device)

        for ii in range(target_len):
            output, h_n = self.decoder(dec_input[:, ii, :], h_n, enc_outputs)
            dec_outputs[:, ii, :] = output.squeeze()

        return dec_outputs

    def predict(self, enc_input, max_len=4):
        batch_size = enc_input.size(0)

        enc_outputs, h_n = self.encoder(enc_input)
        dec_outputs = torch.zeros(batch_size, max_len, self.output_size).to(enc_outputs.device)

        output = enc_input[:, -1, :]

        for ii in range(max_len):
            output, h_n = self.decoder(output.squeeze(), h_n, enc_outputs)
            dec_outputs[:, ii, :] = output.squeeze()

        return dec_outputs
