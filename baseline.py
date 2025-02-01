""" 
Title: Leveraging attention in discourse classification for genre diverse data
Description: Reads rs4 files. Outputs a data frame for processing down the line
Author: Darja Jepifanova, Marco Floess
Date: 2025-02-xx
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, CharacterEmbeddings, StackedEmbeddings
from flair.data import Token, Sentence
from data_manager import load_data
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time
import pickle
import os

glove_embeddings_cache = None


glove_embeddings_cache = None

def load_glove_embeddings(glove_file_path, cache_path="glove_cache.pkl"):
    global glove_embeddings_cache
    #  if cached file exists
    if os.path.exists(cache_path):
        print(f"Loading cached GloVe embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            glove_embeddings_cache = pickle.load(f)
    else:
        print(f"Loading GloVe embeddings from {glove_file_path}...")
        glove_embeddings_cache = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, no_header=True)
        with open(cache_path, "wb") as f:
            pickle.dump(glove_embeddings_cache, f)
    return glove_embeddings_cache

def get_combined_embeddings(batch, glove_embeddings, flair_embeddings, char_embeddings):
    """
    Generate combined token-level embeddings for a batch of sentences using GloVe, FLAIR, and character embeddings.

    Args:
        batch: List of tokenized sentences (list of lists of words).
        glove_embeddings: GloVe embeddings object (Gensim's KeyedVectors or similar).
        flair_embeddings: FLAIR WordEmbeddings object for token-level embeddings.
        char_embeddings: Character embedding object.

    Returns:
        List of tensors, where each tensor represents a sentence's combined embeddings
        (shape: [sentence_length, combined_embedding_dim]).
    """
    combined_embeddings = []

    for sentence in batch:
        token_embeddings = []

        # Create a FLAIR Sentence object for the entire sentence
        flair_sentence = Sentence(" ".join(sentence))

        # Embed the sentence using FLAIR embeddings
        flair_embeddings.embed(flair_sentence)

        for i, word in enumerate(sentence):
            # GloVe embeddings
            if word in glove_embeddings:
                glove_embedding = torch.tensor(glove_embeddings[word], dtype=torch.float)
            else:
                glove_embedding = torch.zeros(300)  # Assuming GloVe embedding dimension is 300

            # FLAIR embeddings (already embedded at the token level)
            flair_embedding = flair_sentence.tokens[i].embedding

            # Character embeddings
            if hasattr(char_embeddings, "get_item_vector"):
                # If character embeddings provide a method to fetch embeddings for a word
                char_embedding = torch.tensor(char_embeddings.get_item_vector(word), dtype=torch.float)
            else:
                # If character embeddings are not available or not implemented
                char_embedding = torch.zeros(100)  # Adjust the dimension as per your character embeddings

            # Combine token embeddings
            combined_token_embedding = torch.cat([glove_embedding, flair_embedding, char_embedding], dim=0)
            token_embeddings.append(combined_token_embedding)

        # Stack all token embeddings for the sentence
        sentence_embedding = torch.stack(token_embeddings, dim=0)
        combined_embeddings.append(sentence_embedding)

    return combined_embeddings

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embeddings):
        outputs, hidden = self.lstm(embeddings)
        outputs = self.dropout(outputs)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_rate=0.3):
        super(Decoder, self).__init__()
        # Define the LSTM as unidirectional (no bidirectional needed)
        self.lstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_outputs):
        # Decoder uses the encoder outputs directly (no need to manage hidden states manually)
        lstm_out, _ = self.lstm(encoder_outputs)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear transformation to output space
        output = self.fc(lstm_out)
        
        return output

class Baseline(nn.Module):
    def __init__(self, glove_embeddings, flair_embeddings, char_embeddings, encoder, decoder):
        super(Baseline, self).__init__()
        self.glove_embeddings = glove_embeddings
        self.flair_embeddings = flair_embeddings
        self.char_embeddings = char_embeddings
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        """
        For each batch, generate combined embeddings using GloVe, FLAIR, and character embeddings,
        and pass them through the encoder-decoder architecture.
        """
        # Generate combined embeddings for the batch using the provided function
        combined_embeddings = get_combined_embeddings(batch, self.glove_embeddings, self.flair_embeddings, self.char_embeddings)

        # Pad sequences to have uniform lengths for batch processing
        padded_embeddings = pad_sequence(combined_embeddings, batch_first=True)  # Shape: [batch_size, max_seq_len, embedding_dim]

        # Pass through Encoder
        encoder_outputs, hidden = self.encoder(padded_embeddings)

        # Pass through Decoder
        probs = self.decoder(encoder_outputs)

        # Take the last output of the sequence (assuming the model is trying to predict a single class label)
        probs = probs[:, -1, :]  # Shape: [batch_size, num_classes]

        return probs

def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    epoch_times = []  # to store epoch durations

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            # Unpack the batch
            sentences = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            # Move labels to the device (ensure dtype is long for CrossEntropyLoss)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = loss_fn(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        val_loss, val_accuracy, _, _, _ = evaluate_model(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_time = time.time() - total_start_time
        estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
        remaining_time = estimated_total - elapsed_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Epoch Time: {epoch_time:.2f} sec, Remaining: {remaining_time/60:.2f} min")

    # Plot validation and training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plot_name = "baseline_val_loss1"
    plt.savefig(f'plots/{plot_name}.png')
    plt.close()


def save_model(model, path="baseline_model.pth"):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="baseline_model.pth"):
    """Load a trained model."""
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            sentences = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            labels = torch.tensor(labels).to(device)
            outputs = model(sentences)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)

    return total_loss / len(dataloader), accuracy, precision, recall, f1, report

def test_model(model, test_dataloader, loss_fn, device):
    """Test the model and print evaluation metrics."""
    test_loss, test_acc, test_prec, test_rec, test_f1, report = evaluate_model(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1-score: {test_f1:.4f}")
    print("\nClassification Report (per label):")
    print(report)

def main():
    embedding_dim = 2448  # Adjust based on the dimensions of GloVe, FLAIR, and character embeddings
    hidden_dim = 128
    output_dim = 32  # Number of classes
    dropout_rate = 0.3
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize embeddings
    glove_embeddings = load_glove_embeddings("glove.42B.300d.txt")  # Load GloVe embeddings (e.g., using Gensim KeyedVectors)
    flair_embeddings = FlairEmbeddings("news-forward")  # Token-level FLAIR embeddings
    char_embeddings = CharacterEmbeddings()  # Load character embeddings (e.g., from AllenNLP or custom)

    # Initialize Encoder and Decoder
    encoder = Encoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)

    # Initialize Baseline model
    model = Baseline(glove_embeddings, flair_embeddings, char_embeddings, encoder, decoder).to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataloader = load_data('data/train', 'relations.csv', batch_size=batch_size)
    val_dataloader = load_data('data/dev', 'relations.csv', batch_size=batch_size)
    test_dataloader = load_data('data/test', 'relations.csv', batch_size=batch_size)
    
    # Train the model
    train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device)

    save_model(model)

    # Load the model for testing
    load_model(model)

    # Evaluate on test data
    test_model(model, test_dataloader, loss_fn, device)

if __name__ == "__main__":
    main()