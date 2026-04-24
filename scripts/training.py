import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import process_data
from dataset import ArticleDataset
from article_classifier_model import ArticleClassifierModel

EPOCHS = 10
LR = 3e-4

df = process_data(pd.read_json("data/mock_articles.json"))

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=67, stratify=df["label"]
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

train_dataset = ArticleDataset(
    train_df["title"].tolist(),
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    tokenizer,
)
val_dataset = ArticleDataset(
    val_df["title"].tolist(),
    val_df["text"].tolist(),
    val_df["label"].tolist(),
    tokenizer,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArticleClassifierModel(bert).to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_correct / train_total:.4f} | "
        f"Val Loss: {val_loss / len(val_loader):.4f} | "
        f"Val Acc: {val_correct / val_total:.4f}"
    )

torch.save(model.classifier.state_dict(), "classifier.pth")

print("\n" + "=" * 50)
print("Evaluation on mock test set")
print("=" * 50)

mock_test_df = process_data(pd.read_json("data/mock_test_articles.json"))

mock_test_dataset = ArticleDataset(
    mock_test_df["title"].tolist(),
    mock_test_df["text"].tolist(),
    mock_test_df["label"].tolist(),
    tokenizer,
)
mock_test_loader = DataLoader(mock_test_dataset, batch_size=16)

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in mock_test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["FAKE", "CREDIBLE"]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\n" + "=" * 50)
print("Evaluation on real test set")
print("=" * 50)

test_df = process_data(pd.read_json("data/test_articles.json"))

test_dataset = ArticleDataset(
    test_df["title"].tolist(),
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    tokenizer,
)
test_loader = DataLoader(test_dataset, batch_size=4)

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["FAKE", "CREDIBLE"]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
