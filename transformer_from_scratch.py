"""
This a Transformer implementation from scratch using PyTorch. Note that all Large Language Models (LLMs) use these
Transformer encoder (e.g. BERT) or decoder (e.g. GPT) blocks for training. Hence, understanding the Transformer in
detail is extremely important.

Note that RNNs process data sequentially (one step at a time) which hinders parallelization whereas Transformers process
the entire sequence in parallel using self-attention mechanisms. RNNs are like reading a book one word at a time,
remembering what came before. Transformers are like reading the whole page at once and deciding what’s important using
attention.

To build our Transformer model, we’ll follow these steps:
1. Import necessary libraries and modules
2. Define the basic building blocks: Multi-Head Attention, Position-wise Feed-Forward Networks, Positional Encoding
3. Build the Encoder and Decoder layers
4. Combine Encoder and Decoder layers to create the complete Transformer model
5. Prepare sample data for sequence-to-sequence task
6. Train the model
"""

# Import necessary libraries and modules
import torch
import torch.nn as nn
import math


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism computes the attention between each pair of elements (e.g. tokens) in a sequence. It
    consists of multiple attention heads that capture different aspects of the input sequence. It allows the model to
    jointly attend to information from different representation subspaces at different positions.

    Instead of performing a single attention function with d_model-dimensional keys, values and queries, it is
    beneficial to linearly project the queries, keys and values h times with different, learned linear projections to
    d_k, d_k and d_v dimensions, respectively. On each of these projected versions of queries, keys and values we then
    perform the attention function in parallel, yielding d_v-dimensional output values. These are concatenated and once
    again projected, resulting in the final values.
    """
    def __init__(self, d_model, num_heads):
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs
                 (default: d_model = 512).
        num_heads: number of heads (default: num_heads = 8).
        """
        super(MultiHeadAttention, self).__init__()

        # Make sure that d_model must be divisible by num_heads!
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        d_k = d_model // num_heads  # d_k is the dimension of queries and keys. NB. d_k * num_heads = d_model.
        self.d_k = d_k
        self.d_v = d_k  # d_v is the dimension of values. Here it is assumed d_k = d_v.

        self.W_q = nn.Linear(d_model, d_model)  # For concatenated step (after scaled dot product attention, see below).
        self.W_k = nn.Linear(d_model, d_model)  # For concatenated step (after scaled dot product attention).
        self.W_v = nn.Linear(d_model, d_model)  # For concatenated step (after scaled dot product attention).
        self.W_o = nn.Linear(self.d_v * self.num_heads, d_model)  # self.d_v * self.num_heads = d_model i.e. d_v = d_k.

    def scaled_dot_product_attention(self, query, key, value, mask):
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Apply a softmax function to obtain the weights (normalized or probability) on the values.
        # attention_scores = attention_scores.softmax(dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)

        # (batch, num_heads, seq_len, seq_len) --> (batch, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_scores, value)  # Same as attention_output = attention_scores @ value

        # Return attention scores as well which can be used for visualization
        return attention_output, attention_scores

    def forward(self, Q, K, V, mask):
        query = self.W_q(Q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.W_k(K)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.W_v(V)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Split d_model dimension into num_heads and d_k
        # (batch, seq_len, d_model) --> (batch, seq_len, num_heads, d_k) --> (batch, num_heads, seq_len, d_k)
        # This allows each attention head to work on its portion of the d_model dimensions separately.
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_v).transpose(1, 2)  # d_v = d_k

        # Calculate attention: computes the weighted sum of values and the attention scores.
        attn_output, attention_scores = self.scaled_dot_product_attention(query, key, value, mask)

        # Combine or concatenate all the heads together.
        # (batch, num_heads, seq_len, d_k) --> (batch, seq_len, num_heads, d_k) --> (batch, seq_len, d_model)
        # contiguous() ensures that the tensor's memory layout is continuous, which can be important for efficiency in
        # certain operations.
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.num_heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model) i.e. the output shape is the same as the input shape.
        output = self.W_o(attn_output)
        return output


# Position-wise Feed-Forward Networks
class PositionWiseFeedForward(nn.Module):
    """
    d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
    d_ff: dimension of the position-wise feedforward network model.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


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
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
        max_seq_length: Maximum length of input sequence.
        """
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
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
        max_seq_length: Maximum length of input sequence.
        """
        super(LearnedPositionalEncoding2, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_seq_length, d_model))
        nn.init.normal_(self.position_embeddings)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.position_embeddings[:seq_len]  # Add positional encoding to token (word) embeddings.


# Encoder Layer
class EncoderLayer(nn.Module):
    """
    An Encoder layer consists of a Multi-Head Attention layer, a Position-wise Feed-Forward layer, and two Layer
    Normalization layers.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
        num_heads: Number of heads (for multi-head attention).
        d_ff: dimension of the position-wise feedforward.
        dropout: dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Multi-head self-attention.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: The input tensor (encoder input).
        mask: Source mask to prevent the model from attending to padding tokens in the source input.
        """
        attn_output = self.self_attn(x, x, x, mask)  # The input x is used to attend to itself. This means that each
        # word in a sentence interacts with every other word in the same sentence.
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Decoder Layer
class DecoderLayer(nn.Module):
    """
    A Decoder layer consists of two Multi-Head Attention layers (masked self-attention & cross-attention), a
    position-wise Feed-Forward layer, and three Layer Normalization layers. These operations enable the decoder to
    generate target sequences based on the input and the encoder output.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        d_model: dimension of the embeddings i.e. the number of expected features in the encoder/decoder inputs.
        num_heads: Number of heads (for multi-head attention).
        d_ff: dimension of the position-wise feedforward.
        dropout: dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Masked multi-head self-attention.
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # Multi-head cross-attention (attending to the
        # encoder's output)y.
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        x: The input tensor (decoder input).
        enc_output: The output from the encoder.
        src_mask: Source mask to prevent the model from attending to padding tokens in the source input.
        tgt_mask: Target mask to prevent the model from attending to future tokens in the target sequence (look-ahead
        mask).
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)  # Masked multi-head self-attention.
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)  # Multi-head cross-attention, between the
        # decoder input (x for Q) and encoder outputs (enc_output for both K & V). The query comes from the decoder,
        # while the key and value come from the encoder.

        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# Transformer Model - Merging it all together:
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff,
                 max_seq_length, dropout, positional_encoding_type):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # Token (word) embedding
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # source token (word) embedding - Categorical
        # variables (dictionary of N words i.e. vocabulary size) are represented by one-hot vectors and then embedded
        # into a linear space, with a size of dimension d_model.
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # target token (word) embedding.

        # Positional encoding
        self.positional_encoding_type = positional_encoding_type
        if self.positional_encoding_type == 'sinusoidal':
            self.positional_encoder = SinusoidalPositionalEncoding(d_model, max_seq_length)
        elif self.positional_encoding_type == 'learned':
            self.positional_encoder = LearnedPositionalEncoding1(d_model, max_seq_length)  # Using nn.Embedding
            # self.positional_encoding = LearnedPositionalEncoding2(d_model, max_seq_length)   # Using nn.Parameter
        else:
            raise ValueError("'Set to correct positional encoding type: 'sinusoidal' for sinusoidal positional encoding"
                             "or 'learned' for learned positional encoding.")

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        # This projection layer is used to convert the high-dimensional vectors (output of the decoder) into logits over
        # the vocabulary.
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        This method creates binary masks for source and target sequences to ignore padding tokens and prevent the
        decoder from attending to future tokens (during training).

        src: the sequence to the encoder.
        tgt: the sequence to the decoder.
        """
        # Padding mask: Prevents from considering padding tokens in the source and target sequences. This is used in
        # both encoder and decoder.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # the additive mask for the src sequence.
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # the additive mask for the tgt sequence. Todo: 2 or 3?
        seq_length = tgt.size(1)
        # Look ahead mask (causal mask): Prevents from not attending to future positions (tokens). This is used in
        # decoder.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Scale the embeddings by the square root of d_model. This is often done to maintain the variance of the
        # embeddings when they are passed through the network, helping with training stability. To read more about
        # this please refer to section 3.4 of the original paper ('Attention is All You Need').
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)

        src_embedded = self.dropout(self.positional_encoder(src))
        tgt_embedded = self.dropout(self.positional_encoder(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


def main():
    # Preparing Sample Data
    src_vocab_size = 2000
    tgt_vocab_size = 2000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_seq_length = 100
    batch_size = 64
    dropout = 0.1
    position_encoding_type = 'learned'   # sinusoidal or learned

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers,
                              num_decoder_layers, d_ff, max_seq_length, dropout, position_encoding_type)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, max_seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, max_seq_length)

    # Training the Model
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])  # tgt_data[:, :-1] is shifted right outputs.
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")


# Execute from the interpreter
if __name__ == "__main__":
    main()
