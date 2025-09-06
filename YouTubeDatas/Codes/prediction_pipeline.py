import torch
import pandas as pd
import numpy as np
from PIL import Image
import joblib 

try:
    from transformers import ViTImageProcessor, ViTModel, BertTokenizer, BertModel
except ImportError:
    print("HATA: 'transformers' kütüphanesi yüklü değil. Lütfen 'pip install transformers' komutuyla yükleyin.")
    exit()

print("Pipeline için tüm modeller ve araçlar yükleniyor...")
try:
    # GÖRSEL İŞLEME 
    VİT_PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    VİT_MODEL = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # METİN İŞLEME
    BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    BERT_MODEL = BertModel.from_pretrained('bert-base-multilingual-cased')

    # EĞİTİLMİŞ YARDIMCI MODELLER VE ARAÇLAR
    PREPROCESSOR = joblib.load('preprocessor_api.pkl') 
    RAW_FEATURE_NAMES = joblib.load('raw_feature_names_api.pkl')
    DEFAULT_VALUES = joblib.load('default_feature_values_api.pkl')
    PCA_MODEL = joblib.load('pca_model.pkl')         
    KMEANS_MODEL = joblib.load('kmeans_model.pkl')   

    print("Pipeline için tüm modeller ve araçlar başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"HATA: Gerekli bir dosya bulunamadı: {e}. Tüm eğitim scriptlerini çalıştırdığınızdan emin olun.")
    exit()
except Exception as e:
    print(f"HATA: Modeller yüklenirken bir sorun oluştu: {e}")
    exit()

# 2. ÖZELLİK ÇIKARMA FONKSİYONLARI

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Bir PIL Image nesnesi alıp 768 boyutlu bir ViT embedding'i döndürür."""
    inputs = VİT_PROCESSOR(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = VİT_MODEL(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def get_text_embedding(title: str) -> np.ndarray:
    """Bir başlık metni alıp 768 boyutlu bir BERT embedding'i döndürür."""
    inputs = BERT_TOKENIZER(title, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# 3. ANA PİPELİNE FONKSİYONU

def create_feature_vector(
    image: Image.Image, 
    title: str,
    sub_count: int,
    view_count: int,
    video_count: int
) -> np.ndarray:
    """
    Görsel ve başlığı alıp, DİNAMİK GÖRSEL ÖZELLİKLERİ HESAPLAYARAK,
    modelin beklediği son özellik vektörünü oluşturur.
    """
    print("Özellik vektörü oluşturuluyor...")
    text_emb = get_text_embedding(title)
    image_emb = get_image_embedding(image)

    pca_values = PCA_MODEL.transform(image_emb)
    cluster_label = KMEANS_MODEL.predict(image_emb)[0]

    print(f"Dinamik görsel özellikleri: PCA1={pca_values[0,0]:.4f}, PCA2={pca_values[0,1]:.4f}, Küme={cluster_label}")

    feature_dict = {}
    
    for i in range(text_emb.shape[1]):
        feature_dict[f'title_emb_{i}'] = text_emb[0, i]
        
    feature_dict['pca1'] = pca_values[0, 0]
    feature_dict['pca2'] = pca_values[0, 1]
    feature_dict['thumbnail_cluster'] = str(cluster_label) # Preprocessor str bekliyor olabilir
    feature_dict['channel_subscriber_count'] = sub_count
    feature_dict['channel_total_view_count'] = view_count
    feature_dict['channel_total_video_count'] = video_count
    
    for feature_name in RAW_FEATURE_NAMES:
        if feature_name not in feature_dict:
            feature_dict[feature_name] = DEFAULT_VALUES.get(feature_name, 0)


    df = pd.DataFrame([feature_dict])
    
    df = df[RAW_FEATURE_NAMES]
    
    print("Ham özellikler oluşturuldu ve sıralandı. Preprocessor'a gönderiliyor...")
    scaled_features = PREPROCESSOR.transform(df)
    
    print(f"Özellik vektörü başarıyla oluşturuldu. Boyut: {scaled_features.shape}")
    return scaled_features