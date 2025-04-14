"""
This code is a complete implementation of a ChatBot (sequence-to-sequence task) using the Transformer architecture: a
transformer built from built-in nn.Transformer of a PyTorch deep learning framework. This Transformer implementation is
based on the original paper 'Attention is All you Need'.

The example given in https://pytorch.org/tutorials/beginner/chatbot_tutorial.html also gives a good ChatBot tutorial
though it uses RNN (particularly GRU with attention) unlike ours which aims to demonstrate how to implement Transformer
using built-in PyTorch framework, for ChatBot example.

Key Advantages of This Implementation:
1. Transformer Architecture: Uses self-attention mechanisms for better context understanding.
2. Positional Encoding: Captures word order information without RNNs.
3. Teacher Forcing: During training, uses ground truth as decoder input.
4. Masking: Properly handles padding and future token masking.
5. Temperature Sampling: Allows for more diverse responses.

Possible Improvements:
1. Larger Dataset: Train on a more extensive conversation dataset.
2. Beam Search: Implement beam search for better response generation.
3. Hyperparameter Tuning: Optimize model size and training parameters.
4. Deployment: Wrap in a web interface using Flask or FastAPI.

This implementation provides a solid foundation for a Transformer-based ChatBot that you can extend with more advanced
features as needed.

For more advanced ChatBot applications, you may fine-tune pre-trained open-source large language models (OS-LLMs) such
as Mistral 7B (most efficient 7B model) or Llama 2 7B/13B (Meta's well-documented models). Fine-tuning an OS-LLM for a
ChatBot application involves adapting a pre-trained model to better suit your specific use case (e.g., customer support,
personal assistant, etc.).
You can choose one of the following fine-tuning methods:
1. Full Fine-Tuning: Updates all model weights, and is best for large, domain-specific datasets.
2. LoRA (Low-Rank Adaptation): Only fine-tunes small adapter layers, and is efficient and works well with limited compute.
3. QLoRA (Quantized LoRA):	LoRA + 4-bit quantization, and is best for low-memory GPUs.
4. RLHF (Reinforcement Learning from Human Feedback): Improves responses via human ratings, and is used for advanced
   ChatBots (e.g., ChatGPT).
Recommendation: Use LoRA or QLoRA (cost-effective, works on consumer GPUs) for fine-tuning an OS-LLM.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # For tokenization
# nltk.download('punkt_tab')

from loguru import logger

from transformer_from_pytorch import Transformer as transformer_pytorch


# Build vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = {}
        self.idx = 4  # Next available index

    def add_sentence(self, sentence):
        for word in word_tokenize(sentence.lower()):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


# Tokenize and numericalize
def sentence_to_tensor(sentence, vocab, device, max_length=15):
    tokens = word_tokenize(sentence.lower())
    indices = [vocab.word2idx.get(token, vocab.word2idx['<unk>']) for token in tokens]
    indices = [vocab.word2idx['<sos>']] + indices + [vocab.word2idx['<eos>']]
    if len(indices) < max_length:
        indices += [vocab.word2idx['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length-1] + [vocab.word2idx['<eos>']]
    return torch.tensor(indices, dtype=torch.long, device=device)


# Create masks
def create_mask(model, vocab, src, tgt, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # Look ahead mask (causal mask) - future token masking
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Padding mask
    src_padding_mask = (src == vocab.word2idx['<pad>'])
    tgt_padding_mask = (tgt == vocab.word2idx['<pad>'])

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# Training Loop
def train_epoch(model, vocab, input_tensors, output_tensors, optimizer, criterion, batch_size, device):
    model.train()
    losses = 0

    # Shuffle the dataset
    indices = list(range(len(input_tensors)))
    random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        src = torch.stack([input_tensors[idx] for idx in batch_indices])
        tgt = torch.stack([output_tensors[idx] for idx in batch_indices])

        # Shift tgt for teacher forcing (uses ground truth as decoder input during training).
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(model, vocab, src, tgt_input, device)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            src, tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # Reshape for loss calculation
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(input_tensors)


# Inference Function
def generate_response(model, input_sentence, vocab, device, max_length=15, temperature=0.7):
    model.eval()

    # Tokenize input
    src = sentence_to_tensor(input_sentence, vocab, device).unsqueeze(0)

    # Initialize target with <sos>
    tgt = torch.tensor([[vocab.word2idx['<sos>']]], device=device)

    with torch.no_grad():
        # Encode the source
        memory = model.transformer.encoder(
            model.positional_encoder(model.encoder_embedding(src) * math.sqrt(model.d_model))
        )

        # Generate tokens one by one
        for i in range(max_length-1):
            # Create target mask
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1), device)

            # Decode
            output = model.transformer.decoder(
                model.positional_encoder(model.decoder_embedding(tgt) * math.sqrt(model.d_model)),
                memory,
                tgt_mask=tgt_mask
            )
            output = model.fc_out(output[:, -1:])

            # Apply temperature
            output = output / temperature
            probs = torch.softmax(output, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs.squeeze(), num_samples=1)
            tgt = torch.cat([tgt, next_token.reshape([1, 1])], dim=1)

            # Stop if <eos> is generated
            if next_token.item() == vocab.word2idx['<eos>']:
                break

    # Convert indices to words
    response = []
    for idx in tgt.squeeze().tolist()[1:]:  # Skip <sos>
        if idx == vocab.word2idx['<eos>']:
            break
        response.append(vocab.idx2word.get(idx, '<unk>'))

    return ' '.join(response)


def main():
    parser = argparse.ArgumentParser(description='ChatBot Example using Transformer Implementation from Scratch.')
    parser.add_argument('--d_model', type=int, default=256,  # default: 512
                        help='Dimension of the embeddings.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of heads.')
    parser.add_argument('--num_encoder_layers', type=int, default=3,  # default: 6
                        help='Number of encoder layers.')
    parser.add_argument('--num_decoder_layers', type=int, default=3,  # default: 6
                        help='Number of decoder layers.')
    parser.add_argument('--dim_feedforward', type=int, default=512,   # default: 2048
                        help='Dimension of the position-wise feedforward network model.')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,  # 32, 100
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,  # 0.001
                        help='Initial learning rate for training')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='Dropout rate to use.')
    parser.add_argument('--positional_encoding_type', type=str, default='sinusoidal',
                        help='positional encoding type to use: sinusoidal or learned.')
    args = parser.parse_args()

    logger.info("Args: {}".format(args))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    # Sample dataset
    qa_pairs = [
        ("Hi", "Hello!"),
        ("How are you?", "I'm good, thanks!"),
        ("What's your name?", "I'm a Transformer chatbot."),
        ("What can you do?", "I can chat with you!"),
        ("Bye", "Goodbye!"),
        ("What time is it?", "I don't have access to the current time."),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ]

    # # Using simple dialogs data from https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot/. Uncomment
    # # this to use it.
    # qa_pairs = []
    # with open("./dialogs.txt", 'r') as f:
    #     for line in f:
    #         line = line.split('\t')
    #         question = line[0]
    #         answer = line[1].split('\n')[0]   # Ignore the \n
    #         qa_pairs.append((question, answer))

    vocab = Vocabulary()
    for q, a in qa_pairs:
        vocab.add_sentence(q)
        vocab.add_sentence(a)

    # Create dataset
    input_tensors = []
    output_tensors = []
    for q, a in qa_pairs:
        input_tensors.append(sentence_to_tensor(q, vocab, device))
        output_tensors.append(sentence_to_tensor(a, vocab, device))

    # Training Setup
    VOCAB_SIZE = len(vocab.word2idx)

    # Initialize model
    model = transformer_pytorch(
        VOCAB_SIZE,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        positional_encoding_type=args.positional_encoding_type
        ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    logger.info("\tTraining has started .....")
    for epoch in range(1, args.num_epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, vocab, input_tensors, output_tensors, optimizer, criterion, args.batch_size,
                                 device)
        end_time = time.time()

        print(f'Epoch: {epoch}/{args.num_epochs}, Train loss: {train_loss:.3f}, Time: {end_time-start_time:.2f}s')
    print(" ---Training is complete----\n")

    # Test the ChatBot
    print("----ChatBot chat testing has started----")
    print("ChatBot: Hi! How can I help you today? (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(model, user_input, vocab, device)
        print(f"Chatbot: {response}")


# Execute from the interpreter
if __name__ == "__main__":
    main()

    # Example Output
    # Chatbot: Hi! How can I help you today? (type 'quit' to exit)
    # You: Hi
    # Chatbot: hello !
    # You: What's your name?
    # Chatbot: i ' m a transformer chatbot .
    # You: Tell me a joke
    # Chatbot: why don't scientists trust atoms ? because they make up everything !
    # You: quit
