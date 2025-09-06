import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

INPUT_DATASET_CSV = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/multimodal_video_analysis_final.csv'
TARGET_COLUMN = 'ilk_10_siraya_ulasti_mi'
FEATURE_IMPORTANCE_FILE = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/xgboost_optimized_all_feature_importances.txt'
IMPORTANCE_THRESHOLD = 0.0 # Bu eşiğin altındaki (veya eşit) özellikler (PCA hariç) çıkarılacak

# Modelde her zaman tutulacak özelliklerin ORİJİNAL adları 
ALWAYS_KEEP_ORIGINAL_FEATURES = ['num__pca1', 'num__pca2']

TEST_SIZE = 0.2
RANDOM_STATE = 42
GRAPH_OUTPUT_DIR = 'grafikler_reduced' 

def parse_feature_importance_file(filepath):
    importances = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f) 
            next(f) 
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    feature_name_transformed = parts[0] 
                    try:
                        importance_score = float(parts[1])
                        importances[feature_name_transformed] = importance_score
                    except ValueError:
                        print(f"Uyarı (parse): '{parts[1]}' skoru float'a çevrilemedi: {line.strip()}")
    except FileNotFoundError:
        print(f"HATA: Özellik önemliliği dosyası bulunamadı: {filepath}")
        return None
    except Exception as e:
        print(f"HATA: Özellik önemliliği dosyası okunurken: {e}")
        return None
    return pd.Series(importances)

def main_evaluate_reduced():
    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True) 
    print(f"--- Azaltılmış Özellik Seti ile XGBoost Modeli Değerlendirmesi ---")
    try:
        df_full = pd.read_csv(INPUT_DATASET_CSV)
        print(f"Veri seti '{INPUT_DATASET_CSV}' yüklendi. Boyut: {df_full.shape}")
    except Exception as e: print(f"HATA: Veri seti okunurken: {e}"); return

    if TARGET_COLUMN not in df_full.columns: print(f"HATA: Hedef değişken '{TARGET_COLUMN}' bulunamadı!"); return
    
    original_len = len(df_full)
    df_full.dropna(subset=[TARGET_COLUMN], inplace=True)
    df_full[TARGET_COLUMN] = df_full[TARGET_COLUMN].astype(int)
    if len(df_full) < original_len: print(f"Hedefte NaN içeren {original_len - len(df_full)} satır çıkarıldı.")
    
    if df_full.empty or df_full[TARGET_COLUMN].nunique() < 2: print("HATA: Modelleme için yetersiz veri/çeşitlilik."); return

    base_features_to_use = [ 
        'total_hours', 'total_rise_steps', 'total_fall_steps',
        'thumbnail_cluster', 'pca1', 'pca2',
        'channel_total_view_count', 'channel_subscriber_count', 'channel_total_video_count'
    ]
    title_emb_cols = [col for col in df_full.columns if col.startswith('title_emb_')]
    all_original_feature_names_for_X = [col for col in (base_features_to_use + title_emb_cols) if col in df_full.columns]
    
    X_original_full = df_full[all_original_feature_names_for_X].copy()
    y = df_full[TARGET_COLUMN]

    numerical_features_full = X_original_full.select_dtypes(include=np.number).columns.tolist()
    categorical_features_full = X_original_full.select_dtypes(include='object').columns.tolist()
    
    if 'thumbnail_cluster' in X_original_full.columns:
        if 'thumbnail_cluster' in numerical_features_full: numerical_features_full.remove('thumbnail_cluster')
        if 'thumbnail_cluster' not in categorical_features_full: categorical_features_full.append('thumbnail_cluster')
        X_original_full.loc[:, 'thumbnail_cluster'] = X_original_full['thumbnail_cluster'].astype(str)

    for col in numerical_features_full:
        if X_original_full[col].isnull().any(): X_original_full.loc[:, col] = X_original_full[col].fillna(X_original_full[col].median())
    for col in categorical_features_full:
        if X_original_full[col].isnull().any(): X_original_full.loc[:, col] = X_original_full[col].fillna(X_original_full[col].mode()[0])

    X_train_original_full, X_test_original_full, y_train, y_test = train_test_split(
        X_original_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=(y if y.nunique() < 10 else None)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_full),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features_full)
        ],
        remainder='drop' 
    )
    preprocessor.fit(X_train_original_full)
    transformed_feature_names = preprocessor.get_feature_names_out() 
    feature_importances_transformed = parse_feature_importance_file(FEATURE_IMPORTANCE_FILE)
    if feature_importances_transformed is None: return
    features_to_keep_transformed = feature_importances_transformed[feature_importances_transformed > IMPORTANCE_THRESHOLD].index.tolist()
    
    # Her zaman tutulacak özelliklerin transform edilmiş adlarını bul ve ekle
    always_keep_transformed = []
    for original_feat_name in ALWAYS_KEEP_ORIGINAL_FEATURES:
        transformed_name_candidate_num = f"num__{original_feat_name}"
        if transformed_name_candidate_num in transformed_feature_names:
            always_keep_transformed.append(transformed_name_candidate_num)

    for feat_to_always_keep in always_keep_transformed:
        if feat_to_always_keep not in features_to_keep_transformed:
            features_to_keep_transformed.append(feat_to_always_keep)
            
    features_to_keep_transformed = sorted(list(set(features_to_keep_transformed))) 

    if not features_to_keep_transformed:
        print("Tutulacak özellik bulunamadı (PCA dahil). Eşik değerini veya özellik önemliliği dosyasını kontrol edin."); return

    print(f"\nAzaltılmış sette tutulacak (transform edilmiş) özellik sayısı: {len(features_to_keep_transformed)}")
    print(f"Tutulacak özelliklerden ilk 5'i: {features_to_keep_transformed[:5]}")

    # Orijinal X_train ve X_test'i transform et
    X_train_transformed_full_array = preprocessor.transform(X_train_original_full)
    X_test_transformed_full_array = preprocessor.transform(X_test_original_full)
    
    # DataFrame'e çevir (indeksleri koruyarak)
    df_X_train_transformed_full = pd.DataFrame(X_train_transformed_full_array, columns=transformed_feature_names, index=X_train_original_full.index)
    df_X_test_transformed_full = pd.DataFrame(X_test_transformed_full_array, columns=transformed_feature_names, index=X_test_original_full.index)

    # Sadece tutulacak transform edilmiş özellikleri seç
    actual_features_to_keep = [f_name for f_name in features_to_keep_transformed if f_name in df_X_train_transformed_full.columns]
    if len(actual_features_to_keep) != len(features_to_keep_transformed):
        print("UYARI: Tutulması istenen bazı transform edilmiş özellikler, transform edilmiş X_train'de bulunamadı!")
        print(f"İstenen: {len(features_to_keep_transformed)}, Bulunan: {len(actual_features_to_keep)}")


    X_train_reduced_transformed = df_X_train_transformed_full[actual_features_to_keep]
    X_test_reduced_transformed = df_X_test_transformed_full[actual_features_to_keep]

    print(f"Azaltılmış (transform edilmiş) eğitim seti boyutu: {X_train_reduced_transformed.shape}")
    if X_train_reduced_transformed.empty: print("HATA: Azaltılmış eğitim seti boş!"); return

    # X_train_reduced_transformed ve X_test_reduced_transformed ile model eğitimi
    sınıf_0_sayisi = y_train.value_counts().get(0, 0)
    sınıf_1_sayisi = y_train.value_counts().get(1, 0)
    scale_pos_weight_val = 1.0
    if sınıf_1_sayisi > 0 and sınıf_0_sayisi > 0:
        scale_pos_weight_val = float(sınıf_0_sayisi / sınıf_1_sayisi)

    # Daha önce bulunan en iyi parametreleri kullan
    best_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
    
    xgboost_reduced_model = XGBClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_val,
        use_label_encoder=False
    )

    print(f"\n--- Azaltılmış Özellik Seti ile XGBoost Modeli Eğitiliyor ---")
    start_time = time.time()
    xgboost_reduced_model.fit(X_train_reduced_transformed, y_train) 
    training_time = time.time() - start_time
    print(f"Eğitim tamamlandı. Süre: {training_time:.2f} saniye.")

    # Değerlendirme
    y_pred_reduced = xgboost_reduced_model.predict(X_test_reduced_transformed)
    y_pred_proba_reduced = xgboost_reduced_model.predict_proba(X_test_reduced_transformed)[:, 1]
    
    roc_auc_reduced = None
    if y_test.nunique() >=2:
        try:
            roc_auc_reduced = roc_auc_score(y_test, y_pred_proba_reduced)
        except ValueError as ve_roc:
            print(f"ROC AUC hesaplanırken hata: {ve_roc}")

    model_name_display_reduced = "XGBoost (Reduced_Features)"
    print(f"\n--- {model_name_display_reduced} Test Seti Performans Metrikleri ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_reduced):.4f}")
    if roc_auc_reduced is not None: print(f"ROC AUC: {roc_auc_reduced:.4f}")
    else: print("ROC AUC: Hesaplanamadı.")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_reduced, zero_division=0))

    # Karışıklık Matrisi
    cm_reduced = confusion_matrix(y_test, y_pred_reduced)
    plt.figure(figsize=(6,4))
    classes_display = y_test.unique().astype(str) 
    classes_display.sort() 

    sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=classes_display, 
                yticklabels=classes_display)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title(f'{model_name_display_reduced} - Karışıklık Matrisi')
    plt.tight_layout()
    cm_filename = os.path.join(GRAPH_OUTPUT_DIR, f'{model_name_display_reduced.lower()}_confusion_matrix.png')
    plt.savefig(cm_filename); plt.close()
    print(f"Karışıklık matrisi '{cm_filename}' olarak kaydedildi.")

    # ROC Eğrisi
    if roc_auc_reduced is not None and y_test.nunique() >= 2:
        fpr_reduced, tpr_reduced, _ = roc_curve(y_test, y_pred_proba_reduced)
        plt.figure(figsize=(6,4))
        plt.plot(fpr_reduced, tpr_reduced, color='orange', label=f'{model_name_display_reduced} (AUC = {roc_auc_reduced:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name_display_reduced} - ROC Eğrisi')
        plt.legend()
        plt.tight_layout()
        roc_filename = os.path.join(GRAPH_OUTPUT_DIR, f'{model_name_display_reduced.lower()}_roc_curve.png')
        plt.savefig(roc_filename); plt.close()
        print(f"ROC Eğrisi '{roc_filename}' olarak kaydedildi.")
    elif y_test.nunique() < 2:
        print("ROC Eğrisi çizilemedi: Test setinde sadece tek bir sınıf var.")


    # Yeni Özellik Önemliliği
    if hasattr(xgboost_reduced_model, 'feature_importances_'):
        importances_reduced = xgboost_reduced_model.feature_importances_
        if len(X_train_reduced_transformed.columns) == len(importances_reduced):
            feature_importances_series_reduced = pd.Series(importances_reduced, index=X_train_reduced_transformed.columns).sort_values(ascending=False)
            print(f"\n{model_name_display_reduced} - Özellik Önemliliği (İlk 50):")
            with pd.option_context('display.max_rows', 55): print(feature_importances_series_reduced.head(50))
            
            # TXT'ye kaydetme
            fi_reduced_txt_filename = os.path.join(GRAPH_OUTPUT_DIR, f'{model_name_display_reduced.lower()}_all_feature_importances.txt')
            with open(fi_reduced_txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"{model_name_display_reduced} - Tüm Özellik Önemlilikleri:\n-------------------------------------------------\n")
                for index, value in feature_importances_series_reduced.items():
                    f.write(f"{index}: {value}\n")
            print(f"Tüm özellik önemlilikleri (azaltılmış) '{fi_reduced_txt_filename}' dosyasına kaydedildi.")

            # Grafik
            plt.figure(figsize=(10, max(10, len(feature_importances_series_reduced.head(50)) // 2) )) 
            feature_importances_series_reduced.head(50).plot(kind='barh')
            plt.title(f'{model_name_display_reduced} - Özellik Önemliliği (İlk 50)')
            plt.gca().invert_yaxis(); plt.tight_layout()
            fi_reduced_png_filename = os.path.join(GRAPH_OUTPUT_DIR, f'{model_name_display_reduced.lower()}_feature_importance_top50.png')
            plt.savefig(fi_reduced_png_filename); plt.close()
            print(f"Özellik önemliliği grafiği (azaltılmış) '{fi_reduced_png_filename}' olarak kaydedildi.")
        else:
            print(f"UYARI: Azaltılmış özellik adı sayısı ({len(X_train_reduced_transformed.columns)}) ile önemlilik skoru sayısı ({len(importances_reduced)}) eşleşmiyor.")

    print(f"\n--- Azaltılmış Özellik Seti ile Değerlendirme Tamamlandı ---")

if __name__ == '__main__':
    main_evaluate_reduced()