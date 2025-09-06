import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np
import re
import csv

DB_FILE = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/SOA-2024-13927/20250228 - db/youtube_with_ai_TR.db'
MODEL_NAME = 'bert-base-multilingual-cased'
TEXT_FIELDS_TO_EMBED = ['title']
EMBEDDING_STRATEGY = 'mean'
OUTPUT_TEXT_FEATURES_CSV = 'textual_features_title.csv'
BATCH_SIZE = 64


def get_device():
    if torch.cuda.is_available():
        print("GPU Kullanılıyor.")
        return torch.device("cuda")
    else:
        print("CPU Kullanılıyor.")
        return torch.device("cpu")


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_text_embeddings(texts, tokenizer, model, device, strategy='mean'):
    if not texts or all(not str(text).strip() for text in texts):
        hidden_size = model.config.hidden_size
        return np.zeros((len(texts), hidden_size)) if texts else np.array([])

    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    if strategy == 'cls':
        embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
    elif strategy == 'mean':
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    else:
        raise ValueError("Geçersiz embedding stratejisi. 'cls' veya 'mean' kullanın.")

    return embeddings

def main_extract_text_features():
    device = get_device()

    print(f"\n--- Veritabanından Metin Verileri Çekiliyor ({DB_FILE}) ---")
    try:
        conn = sqlite3.connect(DB_FILE)
        query = f"SELECT video_id, {TEXT_FIELDS_TO_EMBED[0]} FROM videos"
        df_text_raw = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
        return

    if df_text_raw.empty:
        print("Videolardan metin verisi ('title') çekilemedi.")
        return

    field_to_process = TEXT_FIELDS_TO_EMBED[0]
    df_text_raw[field_to_process] = df_text_raw[field_to_process].apply(preprocess_text)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
    except Exception as e:
        print(f"Hata: Transformer modeli ({MODEL_NAME}) yüklenemedi: {e}")
        return

    texts_to_process = df_text_raw[field_to_process].fillna("").tolist()
    video_ids = df_text_raw['video_id'].tolist()
    hidden_size = model.config.hidden_size

    columns = ["video_id"] + [f"{field_to_process}_emb_{i}" for i in range(hidden_size)]
    with open(OUTPUT_TEXT_FEATURES_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

    print(f"\nEmbedding çıkarma başlıyor (batch size = {BATCH_SIZE})...")
    for i in tqdm(range(0, len(texts_to_process), BATCH_SIZE), desc=f"Embedding: {field_to_process}"):
        batch_texts = texts_to_process[i:i + BATCH_SIZE]
        batch_ids = video_ids[i:i + BATCH_SIZE]

        batch_emb_array = get_text_embeddings(batch_texts, tokenizer, model, device, strategy=EMBEDDING_STRATEGY)

        with open(OUTPUT_TEXT_FEATURES_CSV, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for vid, emb in zip(batch_ids, batch_emb_array):
                writer.writerow([vid] + emb.tolist())

    print(f"\n✅ Metinsel özellikler '{OUTPUT_TEXT_FEATURES_CSV}' dosyasına kaydedildi.")

    del model, tokenizer
    if 'cuda' in str(device):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main_extract_text_features()
