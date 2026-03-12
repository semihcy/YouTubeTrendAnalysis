# YouTube Trend Analysis

This project analyzes the factors influencing the trending performance of YouTube videos using machine learning techniques.  
The study focuses on how **thumbnail images**, **video titles**, and **channel statistics** affect a video's ability to reach high positions in YouTube trending rankings.

The project applies a **multimodal machine learning approach**, combining visual and textual features with video metadata to predict trend success.

---

## 🚀 Project Objective

The goal of this project is to understand how presentation elements such as **thumbnails and titles**, which are the first interaction points for viewers, influence the visibility and success of YouTube videos.

By analyzing these factors, the project aims to provide **data-driven insights into content performance dynamics on YouTube**.

---

## 📊 Dataset

The dataset was collected using the **YouTube Data API v3** and stored in **SQLite databases**.

It includes multiple data sources:

### Video Metadata
- Initial and final trending positions
- Publish date
- Category information
- Channel identifiers

### Time-Series Statistics
- Rank change over time
- Trending duration
- Peak ranking position

### Visual Data
- Video thumbnail images

### Text Data
- Semantic features extracted from video titles

### Channel Statistics
- Total views
- Subscriber count
- Total number of uploaded videos

---

## 🧠 Methodology

Several machine learning and data analysis techniques were applied:

### Visual Feature Extraction
Thumbnail images were processed using a **Vision Transformer (ViT)** model to extract high-level visual representations.

### Text Feature Extraction
Video titles were encoded using a **multilingual BERT model** to capture semantic meaning.

### Dimensionality Reduction & Clustering
- **PCA** was applied to reduce feature dimensionality  
- **K-Means clustering** was used to group thumbnail styles

### Predictive Modeling
An **XGBoost classification model** was trained to predict whether a video would reach a high popularity threshold (e.g., **Top 10 trending position**).

### Hyperparameter Optimization
Model performance was improved using **GridSearchCV**.

---

## 📈 Results

The optimized **XGBoost model** achieved the following performance on the test dataset:

- **Accuracy:** 82.8%  
- **ROC AUC:** 87.1%

The analysis shows that **video titles and channel-level statistics are strong predictors of trending performance**, while thumbnail patterns also contribute to predictive signals.

---

## 🔑 Key Contributions

- Demonstrates a **multimodal machine learning approach** combining visual, textual, and metadata features.
- Provides a **data-driven framework for analyzing YouTube trending dynamics**.
- Highlights how **content presentation elements** (thumbnails and titles) influence performance.

These insights may help **content creators, analysts, and digital strategists** better understand the mechanisms behind trending videos.

---

## 🛠 Technologies Used

- Python  
- XGBoost  
- Vision Transformer (ViT)  
- BERT (Multilingual)  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📫 Contact

- Email: 52semih42@gmail.com  
- LinkedIn:  
  https://www.linkedin.com/in/semih-%C3%A7ay-628945200/
