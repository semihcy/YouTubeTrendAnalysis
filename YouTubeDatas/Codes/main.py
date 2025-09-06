import os
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import io
import xgboost as xgb
import numpy as np

try:
    from prediction_pipeline import create_feature_vector
except ImportError as e:
    print(f"HATA: 'prediction_pipeline.py' dosyası bulunamadı veya içinde hata var: {e}")
    exit()

app = FastAPI(title="YouTube Trend Tahmin API", description="Bir video başlığı ve kapak görseli ile trend olma olasılığını tahmin eder.")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_api_model.json')
    
    print(f"XGBoost modeli şu yoldan yükleniyor: {MODEL_PATH}")

    XGB_MODEL = xgb.Booster()
    XGB_MODEL.load_model(MODEL_PATH) 
    
    print("XGBoost modeli başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: XGBoost modeli yüklenemedi. Dosya adının ve yolunun doğru olduğundan emin ol: {e}")
    exit()


@app.post("/predict/")
async def predict(
    title: str = Form(...), 
    image: UploadFile = File(...),
    subscriber_count: int = Form(...),
    channel_view_count: int = Form(...),
    channel_video_count: int = Form(...)
):
    print(f"\n'/predict' endpoint'ine yeni bir istek geldi. Başlık: '{title}', Dosya: '{image.filename}'")
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"HATA: Görsel dosyası okunamadı: {e}")
        return {"error": f"Geçersiz resim dosyası: {e}"}

    try:
        features = create_feature_vector(
           image=pil_image, 
           title=title,
           sub_count=subscriber_count,
           view_count=channel_view_count,
           video_count=channel_video_count
       )
    except Exception as e:
        print(f"HATA: Özellik çıkarılırken hata oluştu: {e}")
        return {"error": f"Özellik çıkarılırken hata oluştu: {e}"}

    dmatrix_features = xgb.DMatrix(features)
    
    print("XGBoost modeli ile tahmin yapılıyor...")
    prediction_score = XGB_MODEL.predict(dmatrix_features)
    probability = float(prediction_score[0])
    print(f"Tahmin sonucu (olasılık): {probability}")

    return {
        "title": title,
        "filename": image.filename,
        "probability_of_trending": probability,
        "confidence_score": f"{probability:.2%}"
    }