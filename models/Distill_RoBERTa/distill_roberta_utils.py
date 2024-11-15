import os
from tqdm import tqdm

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax

import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import nlpaug.augmenter.word as naw


def get_latest_checkpoint(output_dir):
    # Get all directories starting with 'checkpoint-' in the output directory
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None  # No checkpoints found

    # Sort the checkpoints based on their step numbers (numerical sorting)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

    # Get the latest checkpoint directory
    latest_checkpoint = checkpoints[-1]
    return os.path.join(output_dir, latest_checkpoint)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    
    # Convert logits to probabilities
    probs = softmax(pred.predictions, axis=1)
    
    # Handle binary vs multi-class ROC AUC
    if len(set(labels)) == 2:
        # Binary case, use the positive class probability
        roc_auc = roc_auc_score(labels, probs[:, 1])
    else:
        # Multi-class case, use 'ovr' strategy
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    
    return {"accuracy": acc, "f1": f1, "recall": recall, "precision": precision, "roc_auc": roc_auc}


def balance_and_augment_text_data(df, text_col, target_col, model_path='xlm-roberta-base'):
    # Print class distribution before augmentation
    print("Class distribution before augmentation:")
    print(df[target_col].value_counts())

    # Identify class distribution
    class_counts = df[target_col].value_counts()
    max_count = class_counts.max()
    classes = class_counts.index

    augmented_texts = []
    augmented_labels = []

    # Initialize the augmenter with a multilingual model
    augmenter = naw.ContextualWordEmbsAug(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        action="substitute"
    )

    # Iterate through each class and generate samples to balance the dataset
    for cls in classes:
        cls_data = df[df[target_col] == cls]
        num_samples_to_generate = max_count - class_counts[cls]

        if num_samples_to_generate > 0:
            for _ in range(num_samples_to_generate):
                # Randomly choose a sample to augment
                sample = cls_data.sample(n=1)
                original_text = sample[text_col].values[0]

                # Augment the text
                try:
                    augmented_text = augmenter.augment(original_text)

                    # Handle cases where augmenter returns a list
                    if isinstance(augmented_text, list):
                        augmented_text = augmented_text[0]  # Take the first augmented text

                    # Skip if augmentation fails or results in empty text
                    if not isinstance(augmented_text, str) or not augmented_text.strip():
                        continue

                    augmented_texts.append(augmented_text)
                    augmented_labels.append(cls)

                except Exception as e:
                    print(f"Augmentation error for text '{original_text}': {e}")
                    continue

    # Create a DataFrame for the augmented data
    augmented_df = pd.DataFrame({
        text_col: augmented_texts,
        target_col: augmented_labels
    })

    # Combine original and augmented data
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)

    # Print class distribution after augmentation
    print("\nClass distribution after augmentation:")
    print(df_augmented[target_col].value_counts())

    return df_augmented    


def preprocess_data(data, text_col, target_col, tokenizer, max_length=512):
    texts = data[text_col].tolist()
    labels = data[target_col].tolist()
    encoding = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    encoding["labels"] = labels
    dataset = Dataset.from_dict(encoding)
    return dataset


def fine_tune_roberta(data, text_col, target_col, random_seed=12345, num_labels=2, l2_reg=0.01, num_epochs=30, es_patience=10, resume_checkpoint=None, model_path='final_model', output_dir='results'):
    # Split data into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=random_seed)

    # Load pre-trained tokenizer and model
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Preprocess data
    train_dataset = preprocess_data(train_data, text_col, target_col, tokenizer)
    val_dataset   = preprocess_data(val_data,   text_col, target_col, tokenizer)
    test_dataset  = preprocess_data(test_data,  text_col, target_col, tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"./{output_dir}",
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",        # Save model after each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        weight_decay=l2_reg,
        seed=random_seed,
        load_best_model_at_end=True,  # Load the best model at the end based on the evaluation metric
        metric_for_best_model="eval_loss",  # Specify the metric for choosing the best model
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=es_patience)]  # Early stopping if no improvement in 3 evaluations
    )

    # Fine-tune the model
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Evaluate on test set
    predictions = trainer.predict(test_dataset)
    print(predictions.metrics)

    trainer.save_model(f"./{model_path}")
    tokenizer.save_pretrained(f"./{model_path}")

    torch.cuda.empty_cache()


def predict_with_model_batched(df, text_column, model_path, batch_size=32, max_length=512):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize an empty list to store the predictions
    all_predictions = []
    
    # Iterate over the DataFrame in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df[i:i+batch_size]
        
        # Tokenize the text data
        batch_encodings = tokenizer(batch_df[text_column].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Move input tensors to the same device as the model
        input_ids = batch_encodings["input_ids"].to(device)
        attention_mask = batch_encodings["attention_mask"].to(device)
        
        # Make predictions on the batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.softmax(logits, dim=1).tolist()
        
        all_predictions.extend(batch_predictions)
        
    torch.cuda.empty_cache()
    
    return all_predictions

def plot_train_val_loss(df: pd.DataFrame):
  # Extract the necessary columns for plotting
  epochs = df['Epoch']
  train_loss = df['Training Loss']
  val_loss = df['Validation Loss']
  
  # Plot train and validation loss over epochs
  plt.figure(figsize=(8, 5))
  plt.plot(epochs, train_loss, label='Training Loss', marker='o')
  plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss over Epochs')
  plt.legend()
  plt.grid(True)
  plt.show()  
  