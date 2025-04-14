"""
This is a Transformer implementation using built-in nn.Transformer available in PyTorch deep learning framework for a
ChatBot task i.e. sequence-to-sequence task.
"""
import torch
import torch.nn as nn
import math


# Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    """
    Since transformers do not inherently process tokens in a sequential manner like RNNs (Recurrent Neural Networks),
    they need a way to incorporate the order of tokens. This is achieved through positional encodings, which are vectors
    added to the word embeddings. It is used to inject the position information of each token in the input sequence. It
    uses sine and cosine functions of different frequencies to generate the positional encoding. Though this encoding
    works well on text data, it does not work with image data. So there can be multiple ways of embedding the position
    of an object (text/image ), and they can be fixed or learned during training. This a fixed sinusoidal positional
    encoding.

    Advantages:
        1. No training necessary
        2. No need to know the maximum length in the train set.
        3. Test sequences may have lengths not present in the train set.
    """
    def __init__(self, d_model, max_seq_length=100):
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
        max_seq_length: Maximum length of input sequence.
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # Method 1:
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # (max_seq_length, 1)
        # The denominator div_term is calculated in log space for numerical stability.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # For even indices (0, 2, 4, ..., 2i) for dimension i.
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # For odd indices (1, 3, 5, ..., 2i+1) for dimension i.

        # # Method 2:
        # for pos in range(max_seq_length):
        #     for i in range(d_model):
        #         if i % 2 == 0:   # sin(position * (10000 ** (2i / d_model))
        #             pe[pos][i] = np.sin(pos / (10000 ** (i / d_model)))  # For even indices (0, 2, 4, ..., 2i)
        #         else:   # cos(position * (10000 ** (2i / d_model))
        #             pe[pos][i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))  # For odd indices (1, 3, 5, ..., 2i+1)

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)  # Notice that pe is only a local variable and only gets added to the class
        # using the register_buffer method. This way,the positional encoding become a non-trainable part of the model.

    def forward(self, x):
        # (batch, max_seq_length, d_model)
        return x + self.pe[:, :x.size(1)]  # Add positional encoding to word (token) embeddings. The positional encoding
        # vectors have the same size as the word embeddings i.e. d_model dimensions. Positional Embedding ~= positional
        # encoding + word embedding.


# Learned Positional Encoding - using nn.Embedding
class LearnedPositionalEncoding1(nn.Module):
    """
    Both nn.Embedding and nn.Parameter can be used for learned positional encoding. nn.Embedding uses trainable
    embeddings that can adapt to the specific task. Each position index gets its own trainable vector. nn.Embedding is
    more recommended than nn.Parameter to learn positional encoding since it handles batch dimension automatically and
    for its convenience and standard usage in the PyTorch ecosystem.

    Three issues:
        1. Requires to know the maximum L value among all training sequences.
        2. Some test sequences may have lengths not present in the train set. Hence, it might not generalize as well to
           sequences longer than those seen during training.
        3. It requires more parameters than sinusoidal encoding. It may need more training data to learn effective
           position representations.
    """
    def __init__(self, d_model, max_seq_length=100):
        super(LearnedPositionalEncoding1, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings  # Add positional encoding to token (word) embeddings.


# Learned Positional Encoding - using nn.Parameter
class LearnedPositionalEncoding2(nn.Module):
    """
    Both nn.Embedding and nn.Parameter can be used for learned positional encoding. Both approaches will work and
    learn position embeddings during training, but nn.Embedding is generally preferred for its convenience and standard
    usage in the PyTorch ecosystem. In case of nn.Parameter, you want to manually control every aspect of the operation.
    """
    def __init__(self, d_model, max_seq_length=100):
        super(LearnedPositionalEncoding2, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_seq_length, d_model))
        nn.init.normal_(self.position_embeddings)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.position_embeddings[:seq_len]  # Add positional encoding to token (word) embeddings.


class Transformer(nn.Module):
    """
    Transformer Model Implementation using built-in nn.Transformer PyTorch deep learning framework.
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, positional_encoding_type='sinusoidal'):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)  # word embedding - Categorical variables (dictionary
        # of N words i.e. vocabulary size) are represented by one-hot vectors and then embedded into a linear space.
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding_type = positional_encoding_type
        if self.positional_encoding_type == 'sinusoidal':
            self.positional_encoder = SinusoidalPositionalEncoding(d_model)
        elif self.positional_encoding_type == 'learned':
            self.positional_encoder = LearnedPositionalEncoding1(d_model)  # Using nn.Embedding
            # self.positional_encoder = LearnedPositionalEncoding2(d_model, max_seq_length)   # Using nn.Parameter
        else:
            raise ValueError("'Set to correct positional encoding type: 'sinusoidal' for sinusoidal positional encoding"
                             "or 'learned' for learned positional encoding.")

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
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
        src = self.positional_encoder(src)

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoder(tgt)

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
