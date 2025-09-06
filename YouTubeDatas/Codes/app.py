import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="YouTube Trend Tahminleyici", layout="centered")
st.title("🤖 YouTube Videosu Trend Tahmini")
st.write("Videonuzun başlığını ve kapak görselini yükleyerek, videonun Türkiye trendlerine girme olasılığını modelimizle tahmin edin!")

API_URL = "http://127.0.0.1:8000/predict/"

video_title = st.text_input("Video Başlığı", placeholder="Örn: Sınırsız Et Restoranını Batırdık!")
uploaded_image = st.file_uploader("Kapak Görselini Seçin", type=["jpg", "jpeg", "png"])
st.subheader("Kanal Bilgileri")
subscriber_count = st.number_input(
    "Kanalın Abone Sayısı", 
    min_value=0, 
    value=100000, 
    step=1000
)
channel_view_count = st.number_input(
    "Kanalın Toplam İzlenmesi", 
    min_value=0, 
    value=50000000, 
    step=100000
)
channel_video_count = st.number_input(
    "Kanalın Toplam Video Sayısı", 
    min_value=0, 
    value=500, 
    step=10
)
submit_button = st.button(label='🔮 Tahmin Et')

if submit_button:
    if uploaded_image is not None and video_title:
        image = Image.open(uploaded_image)
        st.image(image, caption="Yüklenen Görsel", use_column_width=True)
        
        with st.spinner("Model, görseli ve metni analiz ediyor... Lütfen bekleyin..."):
            try:
                image_bytes = uploaded_image.getvalue()
                files = {'image': (uploaded_image.name, image_bytes, uploaded_image.type)}
                data = {
                    'title': video_title,
                    'subscriber_count': subscriber_count,
                    'channel_view_count': channel_view_count,
                    'channel_video_count': channel_video_count
                }
                
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Tahmin Başarıyla Tamamlandı!")
                    
                    probability = result.get("probability_of_trending", 0)
                    
                    st.metric(label="İlk 10'a Girme Olasılığı", value=f"{probability:.2%}")
                    st.progress(probability)

                    if probability > 0.65:
                        st.balloons()
                        st.info("🎉 Sonuç: Yüksek Potansiyel! Bu videonun trendlere girme olasılığı oldukça yüksek görünüyor.")
                    elif probability > 0.40:
                        st.warning("🤔 Sonuç: Orta Potansiyel. Videonun trend potansiyeli var ancak başlık veya görselde yapılacak küçük iyileştirmeler şansını artırabilir.")
                    else:
                        st.error("📉 Sonuç: Düşük Potansiyel. Modelimize göre bu videonun trendlere girme olasılığı düşük. Başlığı veya görseli daha dikkat çekici hale getirmeyi düşünebilirsiniz.")
                
                else:
                    error_data = response.json()
                    st.error(f"API Hatası (Kod: {response.status_code}): {error_data.get('detail', response.text)}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Bağlantı Hatası: API sunucusuna bağlanılamadı. Lütfen birinci terminaldeki `uvicorn` sunucusunun çalıştığından emin olun.")
    else:
        st.warning("⚠️ Lütfen hem bir başlık girin hem de bir kapak görseli yükleyin.")