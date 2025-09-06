import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import warnings

warnings.filterwarnings("ignore")

INPUT_DATASET_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/dataset_for_modeling_with_target.csv'
TARGET_COLUMN = 'did_it_reach_the_top10'

FEATURES_TO_USE = [
    'start_position',
    'end_position',
    'channel_total_view_count',
    'channel_subscriber_count',
    'channel_total_video_count',
]

TITLE_EMB_PREFIX = 'title_emb_'
IMAGE_EMB_PREFIX = 'image_emb_'

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3


def main_train_xgboost():
    print(f"--- XGBoost Modeli Eğitimi ve Optimizasyonu Başlatılıyor ---")
    try:
        df = pd.read_csv(INPUT_DATASET_CSV)
        print(f"Veri seti '{INPUT_DATASET_CSV}' yüklendi. Boyut: {df.shape}")
    except FileNotFoundError:
        print(f"HATA: Veri seti dosyası bulunamadı: {INPUT_DATASET_CSV}");
        return
    except Exception as e:
        print(f"HATA: Veri seti okunurken: {e}");
        return

    if TARGET_COLUMN not in df.columns:
        print(f"HATA: Hedef değişken '{TARGET_COLUMN}' veri setinde bulunamadı!");
        return

    original_len = len(df)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    if len(df) < original_len:
        print(f"Hedef değişkende {original_len - len(df)} NaN değerli satır çıkarıldı.")
    print(f"Veri seti boyutu (hedefteki NaN'lar sonrası): {df.shape}")

    if df.empty or df[TARGET_COLUMN].nunique() < 2:
        print("HATA: Modelleme için yeterli veri veya hedef değişken çeşitliliği yok.");
        return

    title_emb_cols = [col for col in df.columns if col.startswith(TITLE_EMB_PREFIX)]
    current_features_to_use = FEATURES_TO_USE + title_emb_cols
    cols_for_X = [col for col in current_features_to_use if col in df.columns]
    if not cols_for_X:
        print(
            "HATA: Model için kullanılacak özellik bulunamadı. FEATURES_TO_USE ve TITLE_EMB_PREFIX ayarlarını kontrol edin.")
        return

    X = df[cols_for_X].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\nKullanılan Özellik Sayısı: {X.shape[1]}")
    print(f"Hedef değişken dağılımı:\n{y.value_counts(normalize=True)}")

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # thumbnail_cluster as a categorical
    if 'thumbnail_cluster' in X.columns:
        if 'thumbnail_cluster' in numerical_features:
            numerical_features.remove('thumbnail_cluster')
        if 'thumbnail_cluster' not in categorical_features:
            categorical_features.append('thumbnail_cluster')
        X.loc[:, 'thumbnail_cluster'] = X['thumbnail_cluster'].astype(str)

    print(f"\nSayısal Özellikler ({len(numerical_features)}): {numerical_features[:6]}...")
    print(f"Kategorik Özellikler ({len(categorical_features)}): {categorical_features}")

    # NaN control
    for col in numerical_features:
        if X[col].isnull().any():
            print(f"-> {col} için NaN sayısı: {X[col].isnull().sum()} (imputer pipeline ile doldurulacak)")
    for col in categorical_features:
        if X[col].isnull().any():
            print(f"-> {col} için NaN sayısı: {X[col].isnull().sum()} (mode ile doldurulacak)")

    # 1) numerical pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2) categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    # 3) Title embedding pipeline
    title_pipeline = None
    title_emb_count = len(title_emb_cols)
    if title_emb_count > 0:
        n_comp_title = min(50, title_emb_count)
        title_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp_title, random_state=RANDOM_STATE))
        ])

    transformers = [
        ('num', numeric_pipeline, numerical_features),
    ]
    if categorical_features:
        transformers.append(('cat', cat_pipeline, categorical_features))
    if title_pipeline is not None:
        transformers.append(('title_emb_pca', title_pipeline, title_emb_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    sınıf_0_sayisi = y.value_counts().get(0, 0)
    sınıf_1_sayisi = y.value_counts().get(1, 0)
    scale_pos_weight_val = 1.0
    if sınıf_1_sayisi > 0 and sınıf_0_sayisi > 0:
        scale_pos_weight_val = float(sınıf_0_sayisi / sınıf_1_sayisi)

    xgboost_base = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_val,
        use_label_encoder=False,
        n_jobs=1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgboost_base)
    ])

    param_grid = {
        'classifier__n_estimators': [45, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    }

    # Train/Test split
    stratify_option = None
    if y.nunique() < 10 and len(y.value_counts()) > 1 and y.value_counts(normalize=True).min() > 0.03:
        stratify_option = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_option
    )
    print(f"\nEğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

    # GridSearchCV
    print(f"\n--- GridSearchCV ile hiperparametre araması (CV={CV_FOLDS}) başlıyor ---")
    grid = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS, scoring='roc_auc', n_jobs=1, verbose=2)
    grid_success = False
    start = time.time()
    try:
        grid.fit(X_train, y_train)
        grid_success = True
        print(f"GridSearch tamamlandı. Süre: {time.time() - start:.1f}s")
        print(f"En iyi CV ROC AUC: {grid.best_score_:.4f}")
        print("En iyi parametreler:", grid.best_params_)
    except Exception as e:
        print(f"GridSearch hata verdi: {e}")
        print("GridSearch başarısız, doğrudan varsayılan pipeline ile devam edilecek.")

    # Best pipeline
    if grid_success:
        best_pipeline = grid.best_estimator_
    else:
        best_pipeline = pipeline
        try:
            best_pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"HATA: Varsayılan pipeline ile fit işlemi de başarısız oldu: {e}")
            return

    # early stopping
    try:
        print("\n--- Best pipeline yeniden fit ediliyor (early_stopping_rounds ile) ---")
        best_pipeline.fit(
            X_train, y_train,
            classifier__eval_set=[(X_test, y_test)],
            classifier__early_stopping_rounds=20,
            classifier__verbose=False
        )
    except Exception as e:
        print(f"Early stopping ile yeniden fit sırasında bir problem oldu: {e}")
        print("Bu durumda en son eldeki best_pipeline kullanılacaktır (early stopping atlanıyor).")


    print("\n--- Test seti üzerinde değerlendirme ---")
    y_pred = best_pipeline.predict(X_test)

    y_pred_proba = None
    try:
        if hasattr(best_pipeline.named_steps['classifier'], "predict_proba"):
            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_proba = None

    roc_auc = None
    if y_pred_proba is not None and y_test.nunique() >= 2:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            print(f"ROC AUC hesaplama hatası: {e}")

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        print("ROC AUC: Hesaplanamadı ya da tek sınıf var.")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    classes_to_display = ['0', '1']
    try:
        if hasattr(best_pipeline.named_steps['classifier'], 'classes_'):
            classes_to_display = best_pipeline.named_steps['classifier'].classes_.astype(str)
    except Exception:
        pass
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_to_display, yticklabels=classes_to_display)
    plt.xlabel('Tahmin Edilen');
    plt.ylabel('Gerçek Değer')
    plt.title('XGBoost - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('xgboost_confusion_matrix.png');
    plt.close()
    print("Confusion matrix 'xgboost_confusion_matrix.png' kaydedildi.")

    # ROC curve
    if roc_auc is not None and y_test.nunique() >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost - ROC Curve');
        plt.legend()
        plt.tight_layout()
        plt.savefig('xgboost_roc_curve.png');
        plt.close()
        print("ROC curve 'xgboost_roc_curve.png' kaydedildi.")
    else:
        print("ROC çizilemedi: y_test tek sınıf içeriyor veya predict_proba yok.")

    # Feature importance
    try:
        clf = best_pipeline.named_steps['classifier']
        pre = best_pipeline.named_steps['preprocessor']
        feature_names = pre.get_feature_names_out()
        importances = clf.feature_importances_
        if len(feature_names) == len(importances):
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            print("\nÖzellik önemliliği (ilk 50):")
            print(fi.head(50))
            fi.head(50).plot(kind='barh', figsize=(10, 12))
            plt.gca().invert_yaxis();
            plt.tight_layout()
            plt.savefig('xgboost_feature_importance_top50.png');
            plt.close()
            print("Feature importance grafiği kaydedildi.")
            fi.to_csv('xgboost_feature_importances.csv')
        else:
            print("UYARI: feature name sayısı ile importance sayısı eşleşmiyor; feature importance gösterilemiyor.")
    except Exception as e:
        print(f"Feature importance alınırken hata: {e}")

    print("\n--- Model ve yardımcı dosyalar kaydediliyor ---")
    try:
        clf = best_pipeline.named_steps['classifier']
        # XGBoost model kaydet
        model_filename = 'xgboost_optimized_model.json'
        clf.save_model(model_filename)
        print(f"XGBoost modeli '{model_filename}' olarak kaydedildi.")

        # Preprocessor (ColumnTransformer) kaydet
        joblib.dump(pre, 'preprocessor.pkl')
        print("Preprocessor 'preprocessor.pkl' olarak kaydedildi.")

        joblib.dump(X.columns.tolist(), 'raw_feature_names.pkl')
        print("Ham özellik isimleri 'raw_feature_names.pkl' olarak kaydedildi.")

    except Exception as e:
        print(f"Model/kaydetme sırasında hata: {e}")

    print("\n--- İşlem tamamlandı ---")


if __name__ == '__main__':
    main_train_xgboost()
