import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(12345)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(12345)


from collections import Counter
import time

class LSTM(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, num_features: int, l1_reg=0.0, l2_reg=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.lstm = nn.LSTM(embedding_dim + num_features, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, embeddings, features):
        # Concatenate embeddings and features
        lstm_input = torch.cat((embeddings, features), dim=-1)
        lstm_input = lstm_input.unsqueeze(1)  # Add a time step dimension
        lstm_outputs, _ = self.lstm(lstm_input)
        lstm_outputs = lstm_outputs[:, -1, :]
        x = self.fc1(lstm_outputs)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def regularization_loss(self):
        l1_loss = sum(param.abs().sum() for param in self.parameters())
        l2_loss = sum(param.pow(2).sum() for param in self.parameters())
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss

  
class DynamicLSTM(nn.Module):
    def __init__(self, embedding_dim: int, num_features: int, hidden_size: int, num_layers: int, num_classes: int,
                 bidirectional: bool = True, dropout: float = 0.5, l1_reg: float = 0.0, l2_reg: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        input_size = embedding_dim + num_features

        self.lstm_forward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                    bidirectional=False, dropout=dropout)
        if bidirectional:
            self.lstm_backward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                         bidirectional=False, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg  

    def forward(self, embeddings, features):
        # Concatenate embeddings and features
        x = torch.cat((embeddings, features), dim=-1)
        x = x.unsqueeze(1)  # Add sequence length dimension
    
        # Forward LSTM
        out_forward, _ = self.lstm_forward(x)
    
        if self.num_directions == 2:
            # Backward LSTM
            x_flipped = torch.flip(x, [1])
            out_backward, _ = self.lstm_backward(x_flipped)
            # Concatenate forward and backward outputs
            out = torch.cat((out_forward[:, -1, :], out_backward[:, 0, :]), dim=1)
        else:
            out = out_forward[:, -1, :]
        
        # Decode the hidden state of the last time step
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
    
        return out

    def regularization_loss(self):
        l1_loss = sum(param.abs().sum() for param in self.parameters())
        l2_loss = sum(param.pow(2).sum() for param in self.parameters())
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss


def preprocess_data(df, embedding_columns, feature_columns, target_column, test_size=0.2, val_size=0.2, random_state=12345, batch_size=16):
    # Split the data into features (embeddings and engineered features) and target
    X_embeddings = df[embedding_columns]
    X_features = df[feature_columns]
    y = df[target_column]

    # Split the data into train, validation, and test sets
    X_train_embeddings, X_test_embeddings, X_train_features, X_test_features, y_train, y_test = train_test_split(
        X_embeddings, X_features, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_embeddings, X_val_embeddings, X_train_features, X_val_features, y_train, y_val = train_test_split(
        X_train_embeddings, X_train_features, y_train, test_size=val_size/(1-test_size),
        random_state=random_state, stratify=y_train
    )

    # Convert the data to tensors
    train_embeddings = torch.tensor(X_train_embeddings.values, dtype=torch.float32)
    train_features = torch.tensor(X_train_features.values, dtype=torch.float32)
    train_labels = torch.tensor(y_train.values, dtype=torch.long)

    val_embeddings = torch.tensor(X_val_embeddings.values, dtype=torch.float32)
    val_features = torch.tensor(X_val_features.values, dtype=torch.float32)
    val_labels = torch.tensor(y_val.values, dtype=torch.long)

    test_embeddings = torch.tensor(X_test_embeddings.values, dtype=torch.float32)
    test_features = torch.tensor(X_test_features.values, dtype=torch.float32)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(train_embeddings, train_features, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_features, val_labels)
    test_dataset = TensorDataset(test_embeddings, test_features, test_labels)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, patience):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    counter = 0

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)  

    start_time = time.time()
    for epoch in range(num_epochs):
        try:
            model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(train_dataloader):
                embeddings = batch[0].to(device)
                features = batch[1].to(device)
                labels = batch[2].to(device)

                optimizer.zero_grad()
                outputs = model(embeddings, features)
                loss = criterion(outputs, labels)
                if hasattr(model, 'regularization_loss'):
                    loss += model.regularization_loss()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_dataloader)
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings = batch[0].to(device)
                    features = batch[1].to(device)
                    labels = batch[2].to(device)

                    outputs = model(embeddings, features)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_dataloader)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        except Exception as e:
            raise e

  
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory / (1024 ** 2)
        print(f"Peak GPU VRAM usage: {peak_memory_mb:.2f} MB")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.show()


def test_model(model, test_dataloader, device):
    model.eval()
    test_preds = []
    test_labels = []
    test_preds_prob = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            embeddings = batch[0].to(device)
            features = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(embeddings, features)
            _, predicted = torch.max(outputs, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_preds_prob.extend(torch.softmax(outputs.detach(), dim=1).cpu().numpy())
    
    test_preds_prob = np.array(test_preds_prob)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Calculate ROC curve and AUC
    num_classes = len(np.unique(test_labels))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels, test_preds_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def add_predictions_to_dataframe(df, model, device, embedding_columns, feature_columns, prediction_column_name='prediction', batch_size=64): 
    model.eval()
    model.to(device)

    # Extract embeddings and features from the DataFrame
    embeddings_np = df[embedding_columns].values
    features_np   = df[feature_columns].values

    # Convert to tensors
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
    features_tensor   = torch.tensor(features_np, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    dataset    = TensorDataset(embeddings_tensor, features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            batch_embeddings = batch[0].to(device)
            batch_features = batch[1].to(device)

            outputs = model(batch_embeddings, batch_features)

            probabilities = torch.softmax(outputs, dim=1)

            _, predicted_classes = torch.max(probabilities, dim=1)

            predictions.extend(predicted_classes.cpu().numpy())

    df[prediction_column_name] = predictions

    return df