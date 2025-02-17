""" 
Title: Enhancing Discourse Relation Classification with Attention Mechanisms on
Genre-Diverse Data
Description: Baseline bidirectinal LSTM model for RST discourse relation 
classification, following the setup by Zeldes&Liu (2020)
Author: Darja Jepifanova, Marco Floess
Date: 2025-02-17
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from flair.embeddings import  FlairEmbeddings, CharacterEmbeddings, StackedEmbeddings
from flair.data import Sentence
from data_manager import load_data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time
import pickle
import os

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

def get_combined_embeddings(batch, glove_embeddings, flair_embeddings, char_embeddings, device):
    combined_embeddings = []

    for sentence in batch:
        token_embeddings = []

        flair_sentence = Sentence(" ".join(sentence))

        flair_embeddings.embed(flair_sentence)

        for i, word in enumerate(sentence):
            # 3. GloVe embeddings
            if word in glove_embeddings.key_to_index:
                glove_embedding = torch.tensor(glove_embeddings[word], dtype=torch.float).to(device)
            else:
                glove_embedding = torch.zeros(300).to(device)

            # 2. FLAIR embeddings 
            flair_embedding = flair_sentence.tokens[i].embedding.to(device)

            # 3. character embeddings
            if hasattr(char_embeddings, "get_item_vector"):
                char_embedding = torch.tensor(char_embeddings.get_item_vector(word), dtype=torch.float).to(device)
            else:
                # unknwon char
                char_embedding = torch.zeros(100).to(device)  

            combined_token_embedding = torch.cat([glove_embedding, flair_embedding, char_embedding], dim=0)
            token_embeddings.append(combined_token_embedding)

        # stack all token embeddings for the sentence
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
        self.lstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_outputs):
        lstm_out, _ = self.lstm(encoder_outputs)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        
        return output

class Baseline(nn.Module):
    def __init__(self, glove_embeddings, flair_embeddings, char_embeddings, encoder, decoder, device):
        super(Baseline, self).__init__()
        self.glove_embeddings = glove_embeddings
        self.flair_embeddings = flair_embeddings
        self.char_embeddings = char_embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, batch):
        # combined embeddings 
        combined_embeddings = get_combined_embeddings(batch, self.glove_embeddings, self.flair_embeddings, self.char_embeddings, self.device)

        # padding
        padded_embeddings = pad_sequence(combined_embeddings, batch_first=True)  

        encoder_outputs, hidden = self.encoder(padded_embeddings)
        probs = self.decoder(encoder_outputs)
        probs = probs[:, -1, :]  

        return probs

def train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    epoch_times = []  

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:

            sentences = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(sentences)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # evaluate on validation set
        val_loss, val_accuracy, _, _, _, _ = evaluate_model(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        elapsed_time = time.time() - total_start_time
        estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
        remaining_time = estimated_total - elapsed_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Epoch Time: {epoch_time:.2f} sec, Remaining: {remaining_time/60:.2f} min")

    # plot validation and training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plot_name = "baseline_val_loss_fine1"
    plt.savefig(f'plots/{plot_name}.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plot_name = "baseline_confusion_fine"
    plt.savefig(f'plots/{plot_name}.png')
    plt.close()

def save_model(model, path="baseline_model_fine1.pth"):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="baseline_model_fine1.pth"):
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
            labels = torch.tensor(labels, dtype=torch.long, device=device)
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
    plot_confusion_matrix(all_labels, all_preds)

    return total_loss / len(dataloader), accuracy, precision, recall, f1, report

def test_model(model, test_dataloader, loss_fn, device):
    """Test the model and print evaluation metrics."""
    test_loss, test_acc, test_prec, test_rec, test_f1, report = evaluate_model(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1-score: {test_f1:.4f}")
    print("\nClassification Report (per label):")
    print(report)

def main():
    embedding_dim = 2448  
    hidden_dim = 128
    output_dim = 32  # number of classes
    dropout_rate = 0.3
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize embeddings
    glove_embeddings = load_glove_embeddings("glove.42B.300d.txt")  
    flair_embeddings = FlairEmbeddings("news-forward")
    char_embeddings = CharacterEmbeddings() 

    # encoder and decoder
    encoder = Encoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)

    model = Baseline(glove_embeddings, flair_embeddings, char_embeddings, encoder, decoder, device).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataloader, _ = load_data('data/train', batch_size=batch_size)
    val_dataloader, _ = load_data('data/dev', batch_size=batch_size)
    considered_clusters = ['cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']
    test_sets = {cluster: load_data(f'data/test', batch_size=batch_size, cluster_group=cluster) for cluster in considered_clusters}

    # Train the model, save it and load for the evaluation
    train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device)
    save_model(model)
    load_model(model)

    print("Evaluating on test set...")
    for cluster, test_data in test_sets.items():
        print(f"\nEvaluating on {cluster}...")
        test_model(model, test_data, loss_fn, device)

if __name__ == "__main__":
    main()