import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class embedding(nn.Module):
    def __init__(self, vocab_size=9000, d_model=512):
        super(embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        return self.emb(ids)


class Projection(nn.Module):
    def __init__(self, hidden_size=512):
        super(Projection, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.W(x)

class transformer(nn.Module):
    def __init__(self, input_len=200, output_len=2, d_model=512, vocab_size=9000):
        super(transformer, self).__init__()
        self.output_len = output_len
        self.input_len = input_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=4)

    def forward(self, x, mask):
        # input - (batch, seq, hidden)
        x = x.permute(1, 0, 2)
        pos_x = self.pos_encoder(x)

        encoder_output = self.encoder(pos_x, src_key_padding_mask=mask)  # (input_len, bs, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)  # torch.transpose(encoder_output, 0, 1)
        return encoder_output


class SMILegacy(nn.Module):
    def __init__(self):
        super(SMILegacy, self).__init__()
        self.enc = torch.load("checkpoints/legacy/enc_auc.pth")
        self.net = torch.load("checkpoints/legacy/net_auc.pth")
        self.proj = torch.load("checkpoints/legacy/proj_auc.pth")
        self.enc.eval()
        self.net.eval()
        self.lsoftmax1 = nn.LogSoftmax()

        self.d_model = self.net.d_model

    def forward(self, context, response, mask_ctx, mask_rsp):
        context_enc = self.enc(context)
        response_enc = self.enc(response)

        c_t = self.net(context_enc, mask_ctx)
        c_t = c_t[:, 0, :]  # torch.mean(c_t, dim=1) #(batch, d)

        r_t = self.net(response_enc, mask_rsp)
        z_t = r_t[:, 0, :]  # torch.mean(z_t, dim=1) # (batch, d)
        z_t = self.proj(z_t)  # (batch, d)

        return c_t, z_t

    def forward_context_only(self, context, attn_mask):
        context_enc = self.enc(context)

        c_t = self.net(context_enc, attn_mask)

        c_t = c_t[:, 0, :]  # torch.mean(c_t, dim=1) #(batch, d)

        return c_t

    def compute_loss(self, c_t, z_t):
        score = torch.mm(c_t, torch.transpose(z_t, 0, 1))  # (batch, batch)
        # batch_size = score.shape[0]
        loss = -torch.mean(torch.diag(self.lsoftmax1(score)))
        # loss = -torch.mean(torch.diag(self.lsoftmax0(score)))
        # loss /= -1. * batch_size  # Take expectation

        return score, loss


class SMIForClassification(nn.Module):
    def __init__(self, num_inputs, num_classes, tokenizer, freeze=True, hidden_size=256, dropout=0.1):
        super(SMIForClassification, self).__init__()
        self.num_classes = num_classes

        device = torch.device("cpu")  # Load to cpu first

        self.cpc = SMILegacy()
        print(f"Loaded legacy model!")

        self.classifier = nn.Sequential(
            nn.Linear(num_inputs*self.cpc.d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        self.freeze = freeze
        if freeze:
            print("** PROBING model.")
            self.cpc.eval()
        else:
            print("** FULL FINETUNE model.")

    def forward(self, context, response=None):
        if self.freeze:
            with torch.no_grad():
                c_t = self.cpc.forward_context_only(context, context == 0)
                if response is not None:
                    z_t = self.cpc.forward_context_only(response, response == 0)
                    c_t = torch.cat([c_t, z_t], dim=-1)
        else:
            c_t = self.cpc.forward_context_only(context, context == 0)
            if response is not None:
                z_t = self.cpc.forward_context_only(response, response == 0)
                c_t = torch.cat([c_t, z_t], dim=-1)

        logits = self.classifier(c_t)
        return logits