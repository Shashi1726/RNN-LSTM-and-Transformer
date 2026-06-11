import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATHS = [
    "/content/drive/MyDrive/goemotions_1.csv",
    "/content/goemotions_1.csv",
    "goemotions_1.csv",
]
SAVE_DIR = "/content/drive/MyDrive/goemotion_models"

MAX_FEATURES = 5000
MAX_VOCAB = 10000
MAX_LEN = 50
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 64

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LEN = 128
TRANSFORMER_EPOCHS = 3
TRANSFORMER_BATCH_SIZE = 8


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_dataset_path():
    for path in DATASET_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find goemotions_1.csv. Checked: " + ", ".join(DATASET_PATHS)
    )


def train_logistic(X_train, X_test, y_train, y_test, label_names):
    print("\nTraining Logistic Regression model...")

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    f1_micro = f1_score(y_test, preds, average="micro")

    print("\nLogistic Regression F1 Micro:", f1_micro)
    print(classification_report(y_test, preds, target_names=label_names, zero_division=0))

    joblib.dump(model, os.path.join(SAVE_DIR, "logistic_model.pkl"))
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl"))

    return f1_micro


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.fc(x)


def train_lstm(X_clean, y_values, label_names):
    print("\nTraining LSTM model...")

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<UNK>")
    tokenizer.fit_on_texts(X_clean)

    X_seq = tokenizer.texts_to_sequences(X_clean)
    X_padded = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded,
        y_values,
        test_size=0.2,
        random_state=42,
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=128,
    )

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    num_classes = len(label_names)

    model = LSTMModel(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=num_classes,
    )

    pos_weight = (y_train_tensor.shape[0] - y_train_tensor.sum(dim=0)) / (
        y_train_tensor.sum(dim=0) + 1e-5
    )
    pos_weight = torch.clamp(pos_weight, max=10)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(LSTM_EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{LSTM_EPOCHS}, Loss: {avg_loss:.4f}")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            all_preds.append(preds)
            all_true.append(y_batch)

    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()

    f1_micro = f1_score(all_true, all_preds, average="micro")
    f1_macro = f1_score(all_true, all_preds, average="macro")

    print("\nLSTM F1 Micro:", f1_micro)
    print("LSTM F1 Macro:", f1_macro)

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_model.pt"))
    joblib.dump(tokenizer, os.path.join(SAVE_DIR, "tokenizer.pkl"))

    return f1_micro, f1_macro, vocab_size


def train_transformer(X_clean, y_values, label_names):
    print("\nTraining Transformer model...")

    from datasets import Dataset, config as datasets_config
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    datasets_config.TORCHVISION_AVAILABLE = False

    transformer_save_path = os.path.join(SAVE_DIR, "transformer_model")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    transformer_df = pd.DataFrame(
        {
            "text": list(X_clean),
            "labels": y_values.astype(float).tolist(),
        }
    )

    train_df, test_df = train_test_split(
        transformer_df,
        test_size=0.2,
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=TRANSFORMER_MAX_LEN,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    remove_train_columns = ["text"]
    remove_test_columns = ["text"]

    if "__index_level_0__" in train_dataset.column_names:
        remove_train_columns.append("__index_level_0__")
    if "__index_level_0__" in test_dataset.column_names:
        remove_test_columns.append("__index_level_0__")

    train_dataset = train_dataset.remove_columns(remove_train_columns)
    test_dataset = test_dataset.remove_columns(remove_test_columns)

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)

        return {
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_DIR, "transformer_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=TRANSFORMER_BATCH_SIZE,
        num_train_epochs=TRANSFORMER_EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(SAVE_DIR, "transformer_logs"),
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    print("\nTransformer Results:")
    print(results)

    model.save_pretrained(transformer_save_path)
    tokenizer.save_pretrained(transformer_save_path)

    return results


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset_path = find_dataset_path()
    print("Using dataset:", dataset_path)

    df = pd.read_csv(dataset_path)
    X = df["text"].astype(str)
    X_clean = X.apply(preprocess)

    # Your previous LSTM file used df.iloc[:, 9:] for labels.
    # If your last column is not an emotion label, change this to df.iloc[:, 9:-1].
    y = df.iloc[:, 9:]
    label_names = y.columns.tolist()

    print("Dataset loaded:", df.shape)
    print("Number of labels:", len(label_names))
    print("Labels:", label_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.2,
        random_state=42,
    )

    logistic_f1_micro = train_logistic(X_train, X_test, y_train, y_test, label_names)
    lstm_f1_micro, lstm_f1_macro, vocab_size = train_lstm(
        X_clean,
        y.values,
        label_names,
    )
    transformer_results = train_transformer(X_clean, y.values, label_names)

    joblib.dump(label_names, os.path.join(SAVE_DIR, "labels.pkl"))

    config = {
        "dataset_path": dataset_path,
        "max_features": MAX_FEATURES,
        "max_vocab": MAX_VOCAB,
        "max_len": MAX_LEN,
        "vocab_size": vocab_size,
        "num_classes": len(label_names),
        "labels": label_names,
        "logistic_f1_micro": float(logistic_f1_micro),
        "lstm_f1_micro": float(lstm_f1_micro),
        "lstm_f1_macro": float(lstm_f1_macro),
        "transformer_model_name": TRANSFORMER_MODEL_NAME,
        "transformer_results": {
            key: float(value)
            for key, value in transformer_results.items()
            if isinstance(value, (int, float, np.number))
        },
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print("\nAll models saved in:")
    print(SAVE_DIR)
    print("\nSaved files:")
    print("logistic_model.pkl")
    print("tfidf_vectorizer.pkl")
    print("lstm_model.pt")
    print("tokenizer.pkl")
    print("labels.pkl")
    print("config.json")
    print("transformer_model/")


if __name__ == "__main__":
    main()
