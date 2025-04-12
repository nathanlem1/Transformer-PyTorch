"""
This is a Transformer implementation using built-in nn.Transformer available in PyTorch deep learning framework for a
ChatBot task i.e. sequence-to-sequence task.

"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding implementation based on sin and cosine functions.
    """
    def __init__(self, d_model, max_len=100):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Uses register_buffer to store pe without making it a learnable parameter.

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    """
    Transformer Model Implementation
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)  # word embedding - Categorical variables (dictionary
        # of N words i.e. vocabulary size) are represented by one-hot vectors and then embedded into a linear space.
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # Embedding and positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Transformer
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Final linear layer
        output = self.fc_out(output)
        return output

