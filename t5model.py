""" 
Title: Enhancing Discourse Relation Classification with Attention Mechanisms on
Genre-Diverse Data
Description: T5-small model for RST discourse relation 
classification
Author: Darja Jepifanova, Marco Floess
Date: 2025-02-17
""" 
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
from data_manager import load_data
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

class T5DiscourseClassifier(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer):
        super(T5DiscourseClassifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_representation)
        return logits

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
            # concatenate text
            texts = [" ".join(tokens) for tokens, _ in batch]   
            labels = torch.tensor([label for _, label in batch], dtype=torch.long).to(device) # Labels as tensor

            # tokenize inputs 
            encoding = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

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

    # loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig('plots/t5_val_loss_fine1.png')
    plt.close()

# confusion matrix for a detailed error analysis
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  xticklabels=True, yticklabels=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plot_name = "t5_confusion"
    plt.savefig(f'plots/{plot_name}.png')
    plt.close()

def save_model(model, path="t5_model_fine1.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="t5_model_fine1.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            texts = [" ".join(tokens) for tokens, _ in batch]  
            labels = torch.tensor([label for _, label in batch], dtype=torch.long).to(device)

            encoding = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
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
    test_loss, test_acc, test_prec, test_rec, test_f1, report = evaluate_model(model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1-score: {test_f1:.4f}")
    print("\nClassification Report (per label):")
    print(report)


def main():
    batch_size = 8
    epochs = 10
    learning_rate = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "t5-small"

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # load data
    train_data = load_data('data/train', batch_size=batch_size)
    val_data = load_data('data/dev', batch_size=batch_size)
    considered_clusters = ['cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']
    test_sets = {cluster: load_data(f'data/test', batch_size=batch_size, cluster_group=cluster) for cluster in considered_clusters}

    num_labels = 32
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5DiscourseClassifier(model_name, num_labels=num_labels, tokenizer=tokenizer).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # train model
    train_model(model, train_data, val_data, loss_fn, optimizer, epochs, device)
    save_model(model, path="t5_model.pth")

    print("Evaluating on test set...")
    for cluster, test_data in test_sets.items():
        print(f"\nEvaluating on {cluster}...")
        test_model(model, test_data, loss_fn, device)

if __name__ == "__main__":
    main()
