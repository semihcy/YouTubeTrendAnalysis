import sqlite3
import pandas as pd
from datetime import datetime

CHANNELS_DB_FILE = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/channels.db' 
OUTPUT_CHANNEL_FEATURES_CSV = 'channel_features.csv'

def calculate_age_in_days(published_at_str, reference_date_str=None):
    if pd.isna(published_at_str):
        return None
    try:
        published_date = pd.to_datetime(published_at_str, errors='coerce')
        if pd.isna(published_date):
            return None

        if reference_date_str:
            reference_date = pd.to_datetime(reference_date_str, errors='coerce')
        else:
            reference_date = datetime.now() 
        if pd.isna(reference_date): 
            return None

        age = (reference_date - published_date).days
        return age if age >= 0 else None 
    except Exception:
        return None


def main_extract_channel_features():
    conn_channels = None
    print(f"\n--- Kanal Veritabanından Özellik Çıkarma Başlatılıyor ({CHANNELS_DB_FILE}) ---")
    try:
        conn_channels = sqlite3.connect(CHANNELS_DB_FILE)
        cursor_channels = conn_channels.cursor()

        df_channels_base = pd.read_sql_query("SELECT channel_id, title AS channel_main_title, published_at AS channel_published_at, country AS channel_country FROM channels", conn_channels)
        print(f"{len(df_channels_base)} kanal için temel bilgi çekildi.")

        if df_channels_base.empty:
            print("Channels tablosunda veri bulunamadı.")
            return

        latest_stats_query = """
        SELECT
            csl.channel_id,
            csl.view_count AS channel_total_view_count,
            csl.subscriber_count AS channel_subscriber_count,
            csl.video_count AS channel_total_video_count,
            csl.timestamp AS channel_stats_timestamp
        FROM channel_statistics_log csl
        INNER JOIN (
            SELECT channel_id, MAX(timestamp) as max_timestamp
            FROM channel_statistics_log
            GROUP BY channel_id
        ) latest ON csl.channel_id = latest.channel_id AND csl.timestamp = latest.max_timestamp;
        """
        df_latest_stats = pd.read_sql_query(latest_stats_query, conn_channels)
        print(f"{len(df_latest_stats)} kanal için en son istatistikler çekildi.")

        df_channel_features = pd.merge(df_channels_base, df_latest_stats, on='channel_id', how='left')

        df_channel_features['channel_age_days'] = df_channel_features['channel_published_at'].apply(calculate_age_in_days)

        df_channel_features['channel_views_per_subscriber'] = (
            df_channel_features['channel_total_view_count'] / df_channel_features['channel_subscriber_count']
        ).replace([float('inf'), -float('inf')], pd.NA) 

        df_channel_features['channel_videos_per_subscriber'] = (
            df_channel_features['channel_total_video_count'] / df_channel_features['channel_subscriber_count']
        ).replace([float('inf'), -float('inf')], pd.NA)

        print("\nÇıkarılan Kanal Özellikleri (İlk 5 Satır):")
        print(df_channel_features.head())

        df_channel_features.to_csv(OUTPUT_CHANNEL_FEATURES_CSV, index=False, encoding='utf-8-sig')
        print(f"\nKanal özellikleri '{OUTPUT_CHANNEL_FEATURES_CSV}' dosyasına kaydedildi.")

    except sqlite3.Error as e:
        print(f"SQLite veritabanı hatası ({CHANNELS_DB_FILE}): {e}")
    except Exception as e:
        print(f"Kanal özellikleri çıkarılırken beklenmedik bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn_channels:
            conn_channels.close()

if __name__ == '__main__':
    main_extract_channel_features()