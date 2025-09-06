import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)

INPUT_DATASET_CSV = os.path.join(PROJECT_ROOT, 'CSV_Files', 'multimodal_video_analysis_final.csv')
TARGET_COLUMN = 'ilk_10_siraya_ulasti_mi'

FEATURES_TO_USE = [
    'thumbnail_cluster', 'pca1', 'pca2',
    'channel_total_view_count', 'channel_subscriber_count', 'channel_total_video_count'
]
TITLE_EMB_PREFIX = 'title_emb_'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 2 

def main_train_xgboost():
    print(f"--- XGBoost Modeli Eğitimi ve Optimizasyonu Başlatılıyor ---")
    try:
        df = pd.read_csv(INPUT_DATASET_CSV)
        print(f"Veri seti '{INPUT_DATASET_CSV}' yüklendi. Boyut: {df.shape}")
    except FileNotFoundError:
        print(f"HATA: Veri seti dosyası bulunamadı: {INPUT_DATASET_CSV}"); return
    except Exception as e:
        print(f"HATA: Veri seti okunurken: {e}"); return

    if TARGET_COLUMN not in df.columns:
        print(f"HATA: Hedef değişken '{TARGET_COLUMN}' veri setinde bulunamadı!"); return

    original_len = len(df)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    if len(df) < original_len:
        print(f"Hedef değişkende {original_len - len(df)} NaN değerli satır çıkarıldı.")
    print(f"Veri seti boyutu (hedefteki NaN'lar sonrası): {df.shape}")

    if df.empty or df[TARGET_COLUMN].nunique() < 2:
        print("HATA: Modelleme için yeterli veri veya hedef değişken çeşitliliği yok."); return

    # Dinamik olarak title_emb sütunlarını bul ve FEATURES_TO_USE'a ekle
    title_emb_cols = [col for col in df.columns if col.startswith(TITLE_EMB_PREFIX)]
    current_features_to_use = FEATURES_TO_USE + title_emb_cols
    cols_for_X = [col for col in current_features_to_use if col in df.columns]
    if not cols_for_X:
        print("HATA: Model için kullanılacak özellik bulunamadı. FEATURES_TO_USE ve TITLE_EMB_PREFIX ayarlarını kontrol edin.")
        return
        
    X = df[cols_for_X].copy() 
    y = df[TARGET_COLUMN]

    print(f"\nKullanılan Özellik Sayısı: {X.shape[1]}")
    print(f"Hedef değişken dağılımı:\n{y.value_counts(normalize=True)}")

    # Kategorik ve sayısal özellikleri X üzerinden belirle
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    if 'thumbnail_cluster' in X.columns:
        if 'thumbnail_cluster' in numerical_features:
            numerical_features.remove('thumbnail_cluster')
        if 'thumbnail_cluster' not in categorical_features:
             categorical_features.append('thumbnail_cluster')
        X.loc[:, 'thumbnail_cluster'] = X['thumbnail_cluster'].astype(str) 

    print(f"\nSayısal Özellikler ({len(numerical_features)} adet): {numerical_features[:3]}...{numerical_features[-3:] if len(numerical_features)>6 else numerical_features[3:]}")
    print(f"Kategorik Özellikler ({len(categorical_features)} adet): {categorical_features}")
    
    # NaN Doldurma
    for col in numerical_features:
        if X[col].isnull().any():
            X.loc[:, col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        if X[col].isnull().any():
            X.loc[:, col] = X[col].fillna(X[col].mode()[0])
    
    if X.isnull().sum().any():
        print("UYARI: Özelliklerde (X) hala NaN değerler var! Bu durum eğitime engel olabilir.")
        print(X.isnull().sum()[X.isnull().sum() > 0])
    else:
        print("Özelliklerdeki (X) NaN değerler dolduruldu.")


    # --- API İÇİN VARSAYILAN DEĞERLERİ KAYDETME ---
    print("\n--- API için varsayılan değerler hesaplanıyor ve kaydediliyor ---")
    try:
        default_values = X[numerical_features].median().to_dict()
        
        for col in categorical_features:
            if not X[col].empty:
                default_values[col] = X[col].mode()[0]
            else:
                default_values[col] = ""
            
        joblib.dump(default_values, 'default_feature_values.pkl')
        print("Varsayılan özellik değerleri 'default_feature_values.pkl' olarak kaydedildi.")
    except Exception as e:
        print(f"HATA: Varsayılan değerler kaydedilirken: {e}")
    # Eğitim ve test setlerine ayırma
    stratify_option = None
    if y.nunique() < 10 and len(y.value_counts()) > 1 and y.value_counts(normalize=True).min() > 0.05:
        stratify_option = y
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_option
    )
    print(f"\nEğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

    # Ön işleme adımları
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )

    # XGBoost için Hiperparametre Arama Uzayı
    param_grid_xgboost = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.05],
        # 'classifier__subsample': [0.7, 0.8],
        # 'classifier__colsample_bytree': [0.7, 0.8] 
    }
    
    sınıf_0_sayisi = y_train.value_counts().get(0, 0)
    sınıf_1_sayisi = y_train.value_counts().get(1, 0)
    scale_pos_weight_val = 1.0 # float olmalı
    if sınıf_1_sayisi > 0 and sınıf_0_sayisi > 0:
        scale_pos_weight_val = float(sınıf_0_sayisi / sınıf_1_sayisi)

    xgboost_base_model = XGBClassifier(random_state=RANDOM_STATE, 
                                       eval_metric='logloss', 
                                       scale_pos_weight=scale_pos_weight_val,
                                       use_label_encoder=False) 
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgboost_base_model)])

    print(f"\n--- XGBoost Modeli için GridSearchCV ile Hiperparametre Optimizasyonu (CV={CV_FOLDS}) ---")
    
    grid_search = GridSearchCV(pipeline, param_grid_xgboost, cv=CV_FOLDS, scoring='roc_auc', verbose=2, n_jobs=-1)
    
    best_xgboost_pipeline = None
    grid_search_successful = False
    start_time = time.time()
    try:
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"GridSearchCV tamamlandı. Süre: {search_time:.2f} saniye.")
        print(f"En iyi ROC AUC (CV): {grid_search.best_score_:.4f}")
        print(f"En iyi parametreler: {grid_search.best_params_}")
        best_xgboost_pipeline = grid_search.best_estimator_
        grid_search_successful = True
    except Exception as e:
        print(f"HATA: GridSearchCV sırasında: {e}")
        print("Varsayılan parametrelerle XGBoost modeli eğitilecek.")

    if not grid_search_successful or best_xgboost_pipeline is None:
        print("GridSearchCV başarısız oldu veya en iyi model bulunamadı. Varsayılan parametrelerle eğitiliyor.")
        best_xgboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                               ('classifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', 
                                                                          scale_pos_weight=scale_pos_weight_val, use_label_encoder=False))])
        start_time_default = time.time()
        try:
            best_xgboost_pipeline.fit(X_train, y_train)
            training_time_default = time.time() - start_time_default
            print(f"XGBoost (varsayılan) eğitimi tamamlandı. Süre: {training_time_default:.2f} saniye.")
        except Exception as e_fit:
            print(f"HATA: XGBoost (varsayılan) eğitimi sırasında: {e_fit}")
            return

    if best_xgboost_pipeline is None:
        print("HATA: XGBoost modeli eğitilemedi.")
        return

    model_name_display = "XGBoost (Optimized)" if grid_search_successful else "XGBoost (Default)"

    print(f"\n{model_name_display} - Test seti üzerinde tahmin yapılıyor...")
    y_pred = best_xgboost_pipeline.predict(X_test)
    
    roc_auc = None
    y_pred_proba_for_roc = y_pred # Varsayılan

    if hasattr(best_xgboost_pipeline.named_steps['classifier'], "predict_proba"):
        y_pred_proba = best_xgboost_pipeline.predict_proba(X_test)[:, 1]
        y_pred_proba_for_roc = y_pred_proba
        try:
            if y_test.nunique() >= 2:
                 roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                 print("Test setindeki hedef değişken sadece tek bir sınıf içeriyor. ROC AUC tanımsız.")
        except ValueError as ve:
            print(f"ROC AUC hesaplanırken hata (predict_proba): {ve}.")
            
    print(f"\n--- {model_name_display} Test Seti Performans Metrikleri ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: Hesaplanamadı.")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Karışıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    classes_to_display = ['0','1'] 
    if hasattr(best_xgboost_pipeline.named_steps['classifier'], 'classes_'):
        classes_to_display = best_xgboost_pipeline.named_steps['classifier'].classes_.astype(str)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_to_display, 
                yticklabels=classes_to_display)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title(f'{model_name_display} - Karışıklık Matrisi')
    plt.tight_layout()
    file_suffix_cm = f'{model_name_display.replace(" ", "_").lower().replace("(", "").replace(")", "")}_confusion_matrix.png'
    plt.savefig(file_suffix_cm)
    plt.close()
    print(f"Karışıklık matrisi '{file_suffix_cm}' olarak kaydedildi.")

    # ROC Eğrisi
    if roc_auc is not None and y_test.nunique() >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_for_roc)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{model_name_display} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name_display} - ROC Eğrisi')
        plt.legend()
        plt.tight_layout()
        file_suffix_roc = f'{model_name_display.replace(" ", "_").lower().replace("(", "").replace(")", "")}_roc_curve.png'
        plt.savefig(file_suffix_roc)
        plt.close()
        print(f"ROC Eğrisi '{file_suffix_roc}' olarak kaydedildi.")
    elif y_test.nunique() < 2:
        print("ROC Eğrisi çizilemedi: Test setinde sadece tek bir sınıf var.")

    # Özellik Önemliliği
    if hasattr(best_xgboost_pipeline.named_steps['classifier'], 'feature_importances_'):
        try:
            feature_names_out = best_xgboost_pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = best_xgboost_pipeline.named_steps['classifier'].feature_importances_
            
            if len(feature_names_out) == len(importances):
                feature_importances_series = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)
                
                print(f"\n{model_name_display} - Özellik Önemliliği (İlk 50):")
                with pd.option_context('display.max_rows', 55, 'display.max_columns', None, 'display.width', 1000):
                    print(feature_importances_series.head(50))
                    
                    all_feature_importance_filename = f'{model_name_display.replace(" ", "_").lower().replace("(", "").replace(")", "")}_all_feature_importances.txt'
                    with open(all_feature_importance_filename, 'w', encoding='utf-8') as f:
                        f.write(f"{model_name_display} - Tüm Özellik Önemlilikleri:\n")
                        f.write("-------------------------------------------------\n")
                        for index, value in feature_importances_series.items():
                            f.write(f"{index}: {value}\n")
                    print(f"Tüm özellik önemlilikleri '{all_feature_importance_filename}' dosyasına kaydedildi.")

                plt.figure(figsize=(10, 14))
                feature_importances_series.head(50).plot(kind='barh')
                plt.title(f'{model_name_display} - Özellik Önemliliği (İlk 50)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                file_suffix_fi = f'{model_name_display.replace(" ", "_").lower().replace("(", "").replace(")", "")}_feature_importance_top50.png'
                plt.savefig(file_suffix_fi)
                plt.close()
                print(f"Özellik önemliliği grafiği '{file_suffix_fi}' olarak kaydedildi.")
            else:
                print(f"UYARI: Özellik adı sayısı ({len(feature_names_out)}) ile önemlilik skoru sayısı ({len(importances)}) eşleşmiyor. Özellik önemliliği gösterilemiyor.")
        except Exception as e:
            print(f"Özellik önemliliği alınırken veya çizilirken hata: {e}")
    else:
        print(f"{model_name_display} için 'feature_importances_' özelliği bulunamadı.")
    
    print(f"\n--- XGBoost Modeli İşlemi Tamamlandı ---")
    
    print("\n--- Model ve Yardımcı Dosyalar Kaydediliyor ---")

    try:
        best_model_to_save = best_xgboost_pipeline.named_steps['classifier']
        model_filename = 'xgboost_api_model.json'
        best_model_to_save.save_model(model_filename)
        print(f"En iyi XGBoost modeli '{model_filename}' olarak başarıyla kaydedildi.")

        preprocessor_to_save = best_xgboost_pipeline.named_steps['preprocessor']
        preprocessor_filename = 'preprocessor_api.pkl'
        joblib.dump(preprocessor_to_save, preprocessor_filename)
        print(f"Fit edilmiş Preprocessor (ColumnTransformer) '{preprocessor_filename}' olarak kaydedildi.")

        raw_feature_names = X.columns.tolist()
        feature_list_filename = 'raw_feature_names_api.pkl'
        joblib.dump(raw_feature_names, feature_list_filename)
        print(f"Ham özellik listesi '{feature_list_filename}' olarak kaydedildi.")
        
        default_values_filename = 'default_feature_values_api.pkl' # <-- Adı değiştir
        joblib.dump(default_values, default_values_filename)
        print(f"API varsayılan değerleri '{default_values_filename}' olarak kaydedildi.")
  
    except Exception as e:
        print(f"HATA: Model veya yardımcı dosyalar kaydedilirken bir sorun oluştu: {e}")
    
    print(f"\n--- XGBoost Modeli İşlemi Tamamlandı ---")

if __name__ == '__main__':
    main_train_xgboost()
