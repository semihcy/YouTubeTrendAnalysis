# YouTube Trend Analysis

Bu proje, YouTube videolarının trend performansını etkileyen faktörleri incelemeyi amaçlamaktadır. Özellikle, videoların **thumbnail görselleri**, **başlıkları** ve **kanal istatistiklerinin** trend sıralamasına etkisi makine öğrenmesi teknikleriyle analiz edilmiştir.

---

## 🚀 Projenin Amacı
YouTube'da yüksek görünürlük ve başarı için kritik olan trend performansının, kullanıcıların içerikle ilk temas noktası olan görsel ve metinsel unsurlardan nasıl etkilendiğini anlamak.

---

## 📊 Kullanılan Veri Seti

- **Video meta verileri:** Başlangıç ve bitiş pozisyonları, yayın tarihi, kategori ve kanal bilgileri.  
- **Zaman serisi istatistikleri:** Video sıralama değişimleri, trend süresi, zirve pozisyonu.  
- **Görseller:** Videoların thumbnail görselleri.  
- **Metin:** Video başlıklarından çıkarılan anlamsal özellikler.  
- **Kanal bilgileri:** Kanalın toplam görüntüleme sayısı, abone sayısı, toplam video sayısı.  

Tüm veriler **YouTube Data API v3** ve **SQLite veritabanları** üzerinden elde edilmiştir.

---

## 🧠 Kullanılan Yöntemler

- **Görsel Özellikler:** Vision Transformer (ViT) ile thumbnail görsellerinden çıkarılmıştır.  
- **Metin Özellikleri:** Çok dilli BERT modeli ile video başlıklarından çıkarılmıştır.  
- **Boyut İndirgeme ve Kümeleme:** PCA ve K-Means kullanılarak görsel thumbnail kümeleri oluşturulmuştur.  
- **Makine Öğrenmesi:** XGBoost sınıflandırma modeli ile videoların belirli bir zirve popülerlik eşiğine (örneğin ilk 10 sıra) ulaşıp ulaşmadığı tahmin edilmiştir.  
- **Hiperparametre Optimizasyonu:** GridSearchCV kullanılmıştır.

---

## 📈 Sonuçlar

Optimize edilmiş XGBoost modeli, test verisinde:  

- **Doğruluk:** %82.8  
- **ROC AUC:** %87.1  

Analiz, özellikle **video başlığı** ve **kanal istatistiklerinin**, trend performansını tahmin etmede güçlü belirleyiciler olduğunu göstermiştir.

---

## 🔑 Anahtar Kelimeler

YouTube, Trend Analizi, Makine Öğrenmesi, XGBoost, Vision Transformer (ViT), BERT, Çok Modlu Analiz

---

## 💡 Katkılar

Bu çalışma, thumbnail ve başlık gibi kolayca manipüle edilebilir sunum öğelerinin trend başarısı üzerindeki potansiyel etkisini **nicel olarak ortaya koymaktadır**. İçerik üreticiler ve veri analistleri için **YouTube içerik dinamiklerini anlamada değerli içgörüler** sağlamaktadır.

---

## 🛠 Kullanılan Teknolojiler
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF9900?style=flat)
![BERT](https://img.shields.io/badge/BERT-2CA5E0?style=flat)
![Vision Transformer](https://img.shields.io/badge/ViT-6C63FF?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AADD?style=flat)

---

## 📫 İletişim
- Email: 52semih42@gmail.com  
- LinkedIn: [Semih ÇAY](https://www.linkedin.com/in/semih-%C3%A7ay-628945200/)

---

> “Veri sadece sayılar değildir, doğru analiz edildiğinde hikaye anlatır.”  
