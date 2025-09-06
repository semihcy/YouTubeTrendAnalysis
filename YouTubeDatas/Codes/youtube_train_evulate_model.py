import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

INPUT_DATASET_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/dataset_for_modeling_with_target.csv'
TARGET_COLUMN = 'did_it_reach_the_top10'

BASE_FEATURES = [
    'channel_total_view_count',
    'channel_subscriber_count',
    'channel_total_video_count'
]
TITLE_EMB_PREFIX = 'title_emb_'
IMAGE_EMB_PREFIX = 'image_emb_'

TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_evaluate_single_model(X_train, X_test, y_train, y_test, model, model_name, preprocessor, n_components=100):
    """Modeli eğitip değerlendirir"""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('classifier', model)
    ])

    print(f"\n--- {model_name} Modeli Eğitiliyor ---")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"{model_name} eğitimi tamamlandı. Süre: {training_time:.2f} saniye.")

    # tahminler
    y_pred = pipeline.predict(X_test)
    y_pred_proba = (pipeline.predict_proba(X_test)[:, 1]
                    if hasattr(pipeline.named_steps['classifier'], "predict_proba")
                    else y_pred)

    print(f"\n--- {model_name} Test Seti Performans Metrikleri ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title(f'{model_name} - Karışıklık Matrisi')
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()

    return pipeline


def main():
    df = pd.read_csv(INPUT_DATASET_CSV)
    print(f"Veri seti '{INPUT_DATASET_CSV}' yüklendi. Boyut: {df.shape}")

    # hedef kolon
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # özellik kolonları
    title_emb_cols = [c for c in df.columns if c.startswith(TITLE_EMB_PREFIX)]
    image_emb_cols = [c for c in df.columns if c.startswith(IMAGE_EMB_PREFIX)]
    features = BASE_FEATURES + title_emb_cols + image_emb_cols

    X = df[features].copy()
    y = df[TARGET_COLUMN]

    # sayısal ve kategorik kolonlar
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # NaN doldurma
    for col in numerical_features:
        X.loc[:, col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        X.loc[:, col] = X[col].fillna(X[col].mode()[0])

    # train-test split
    stratify_option = y if y.nunique() < 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_option
    )

    print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False),
             categorical_features)
        ],
        remainder='passthrough'  # PCA içinde embeddings işlenecek
    )

    # Modeller
    models_to_try = {
        "Logistic Regression": LogisticRegression(
            solver='liblinear', random_state=RANDOM_STATE,
            class_weight='balanced', max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, n_estimators=350,
            learning_rate=0.08, use_label_encoder=False, eval_metric='logloss',
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))
    }

    trained_pipelines = {}
    for model_name, model_instance in models_to_try.items():
        trained_pipeline = train_evaluate_single_model(
            X_train, X_test, y_train, y_test,
            model_instance, model_name, preprocessor,
            n_components=45
        )
        trained_pipelines[model_name] = trained_pipeline


if __name__ == '__main__':
    main()
