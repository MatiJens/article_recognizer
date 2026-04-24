import pandas as pd
import re

with open("data/stop_words.txt") as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())


def _remove_stop_words(text: str) -> str:
    return " ".join(w for w in text.split() if w.lower() not in STOP_WORDS)


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "<NUM>", text)
    text = re.sub(r"[^a-z\s<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    for col in ["title", "text"]:
        data[col] = data[col].apply(_remove_stop_words)
        data[col] = data[col].apply(_clean_text)
    return data
