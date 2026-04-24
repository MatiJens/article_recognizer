import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from scripts.article_classifier_model import ArticleClassifierModel
from scripts.data_preprocessing import process_data
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

model = ArticleClassifierModel(bert).to(device)
model.classifier.load_state_dict(torch.load("classifier.pth", map_location=device))
model.eval()

LABELS = {0: "FAKE", 1: "CREDIBLE"}


def predict(articles: pd.DataFrame) -> list[dict]:
    articles = process_data(articles)

    encoding = tokenizer(
        articles["title"].tolist(),
        articles["text"].tolist(),
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    results = []
    for i in range(len(probs)):
        predicted = probs[i].argmax().item()
        results.append(
            {
                "label": LABELS[predicted],
                "confidence": f"{probs[i][predicted].item() * 100:.2f}%",
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str)
    parser.add_argument("--title", type=str)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()

    if args.json:
        articles = pd.read_json(args.json)
        results = predict(articles)
    elif args.title and args.text:
        articles = pd.DataFrame([{"title": args.title, "text": args.text}])
        results = predict(articles)
    else:
        print("Pass --json or --title and --text flags.")
        exit(1)

    for i, result in enumerate(results):
        print(f"\nArticle {i + 1}: {result['label']} ({result['confidence']})")
