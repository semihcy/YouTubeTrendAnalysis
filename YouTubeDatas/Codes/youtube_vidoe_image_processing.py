import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


POSITION_CSV_FILE = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/position_changes.csv'
THUMBNAIL_DIR = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/thumbnails_tr'
OUTPUT_ANALYSIS_FILE = '../CSV_Files/multimodal_analysis_with_no_pca.csv'
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
N_CLUSTERS = 10


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def extract_features(image_path, processor, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    except Exception:
        return None


def analyze_video_trends(group):
    first_record = group.iloc[0]
    last_record = group.iloc[-1]
    return {
        'start_position': first_record.get('previous_position', first_record.get('new_position')),
        'end_position': last_record.get('new_position')
    }


def main():
    device = get_device()

    print("Modeller ve veriler yükleniyor...")
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    model = ViTModel.from_pretrained(VIT_MODEL_NAME).to(device)
    model.eval()
    df_pos = pd.read_csv(POSITION_CSV_FILE, parse_dates=['change_timestamp', 'previous_timestamp'])

    # Adım 3: Ham Görsel Özelliklerini Çıkar
    print("Thumbnail'lerden ham 768 boyutlu özellikler çıkarılıyor...")
    unique_video_ids = df_pos['video_id'].unique()
    video_features = {}
    for video_id in tqdm(unique_video_ids, desc="Görsel Özellik Çıkarımı"):
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{video_id}.jpg")
        features = extract_features(thumbnail_path, processor, model, device)
        if features is not None:
            video_features[video_id] = features

    # Adım 4: Trend Dinamiklerini Hesapla ve Özelliklerle Birleştir
    print("Trend dinamikleri hesaplanıyor ve özelliklerle birleştiriliyor...")
    trend_analysis_results = []
    for video_id, group in tqdm(df_pos.groupby('video_id'), desc="Trend Analizi"):
        trend_summary = analyze_video_trends(group)  # Bu fonksiyonun içini doldurmayı unutma
        if trend_summary and video_id in video_features:
            trend_summary['video_id'] = video_id
            trend_summary['title'] = group['title'].iloc[0]
            trend_summary['features'] = video_features[video_id]
            trend_analysis_results.append(trend_summary)

    df_analysis = pd.DataFrame(trend_analysis_results)
    print(f"{len(df_analysis)} video için analiz tamamlandı.")

    if df_analysis.empty:
        print("Analiz için geçerli video bulunamadı.")
        return

    print("Nihai CSV dosyası ham 768 boyutlu görsel özelliklerle oluşturuluyor...")

    image_embeddings_df = pd.DataFrame(
        df_analysis['features'].tolist(),
        index=df_analysis.index
    )
    image_embeddings_df.columns = [f'image_emb_{i}' for i in range(image_embeddings_df.shape[1])]

    df_final_to_save = df_analysis.drop(columns=['features'])
    df_final_to_save = pd.concat([df_final_to_save, image_embeddings_df], axis=1)

    try:
        df_final_to_save.to_csv(OUTPUT_ANALYSIS_FILE, index=False, encoding='utf-8-sig')
        print(f"Kaydetme tamamlandı: '{OUTPUT_ANALYSIS_FILE}'. Toplam sütun sayısı: {len(df_final_to_save.columns)}")
    except Exception as e:
        print(f"Hata: Sonuçlar CSV'ye kaydedilemedi: {e}")


if __name__ == "__main__":
    main()