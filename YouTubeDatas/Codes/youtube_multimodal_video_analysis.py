import pandas as pd
import sqlite3 

VISUAL_ANALYSIS_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/multimodal_analysis_with_no_pca.csv'
TEXT_FEATURES_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/textual_features_title.csv'
CHANNEL_FEATURES_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/channel_features.csv'
VIDEOS_DB_FILE = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/SOA-2024-13927/20250228 - db/youtube_with_ai_TR.db'
OUTPUT_FINAL_MERGED_CSV = 'multimodal_video_analysis_final_with_channel.csv'

def main_merge_all_features():
    print(f"\n--- Tüm Özellikleri Birleştirme Başlatılıyor ---")
    try:
        df_visual = pd.read_csv(VISUAL_ANALYSIS_CSV)
        print(f"Görsel analiz dosyası '{VISUAL_ANALYSIS_CSV}' yüklendi ({len(df_visual)} kayıt).")

        df_text = pd.read_csv(TEXT_FEATURES_CSV)
        print(f"Metinsel özellikler dosyası '{TEXT_FEATURES_CSV}' yüklendi ({len(df_text)} kayıt).")

        df_channel = pd.read_csv(CHANNEL_FEATURES_CSV)
        print(f"Kanal özellikleri dosyası '{CHANNEL_FEATURES_CSV}' yüklendi ({len(df_channel)} kayıt).")

        conn_videos = None
        try:
            conn_videos = sqlite3.connect(VIDEOS_DB_FILE)
            df_video_channel_map = pd.read_sql_query("SELECT video_id, channel_id, category_id, published_at AS video_published_at FROM videos", conn_videos)
            print(f"Videos tablosundan video-kanal eşleştirmesi çekildi ({len(df_video_channel_map)} kayıt).")
        except Exception as e:
            print(f"HATA: Videos tablosundan veri çekilemedi: {e}")
            return
        finally:
            if conn_videos:
                conn_videos.close()

        #Görsel ve Metinsel verileri birleştir
        df_merged_vt = pd.merge(df_visual, df_text, on='video_id', how='inner')
        print(f"Görsel ve metinsel veriler birleştirildi. Ara toplam: {len(df_merged_vt)} video.")

        #Bu birleşik veriyi video-kanal eşleştirmesiyle birleştir
        df_merged_vtc_map = pd.merge(df_merged_vt, df_video_channel_map, on='video_id', how='inner')
        print(f"Video-Kanal eşleştirmesi eklendi. Ara toplam: {len(df_merged_vtc_map)} video.")

        #Kanal özelliklerini channel_id üzerinden ekle
        df_final_merged = pd.merge(df_merged_vtc_map, df_channel, on='channel_id', how='left')
        print(f"Kanal özellikleri eklendi. Nihai toplam: {len(df_final_merged)} video.")

        print("\nBirleştirilmiş nihai verideki eksik değerler (ilk 20 sütun):")
        print(df_final_merged.isnull().sum().head(20))

        df_final_merged.to_csv(OUTPUT_FINAL_MERGED_CSV, index=False, encoding='utf-8-sig')
        print(f"\nNihai birleştirilmiş çok modlu analiz verisi (kanal bilgileri dahil) '{OUTPUT_FINAL_MERGED_CSV}' dosyasına kaydedildi.")

    except FileNotFoundError as e:
        print(f"HATA: Gerekli CSV dosyalarından biri bulunamadı. {e}")
    except Exception as e:
        print(f"Veri birleştirme sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_merge_all_features()