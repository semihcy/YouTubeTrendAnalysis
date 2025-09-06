import sqlite3
import os
import pandas as pd 

DB_FILE = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/20250228 - db/youtube_with_ai_TR.db'  
THUMBNAIL_DIR = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/thumbnails_tr/'        
OUTPUT_CSV_FILE = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/position_changes.csv' 

def analyze_video_positions(db_path, thumbnail_dir):

    conn = None
    print(f"Veritabanına bağlanılıyor: {db_path}")
    try:
        conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        query = """
        SELECT
            vs.video_id,
            v.title, 
            vs.current_position,
            vs.updated_at 
        FROM
            video_stats vs
        LEFT JOIN
            videos v ON vs.video_id = v.video_id
        ORDER BY
            vs.video_id,       
            vs.updated_at;    
        """
        print("Veri çekiliyor...")
        df = pd.read_sql_query(query, conn, parse_dates=['updated_at'])
        print(f"Toplam {len(df)} kayıt çekildi.")

        if df.empty:
            print("video_stats tablosunda veri bulunamadı.")
            return

        print("Pozisyon değişiklikleri analiz ediliyor...")

        results = []

        for video_id, group in df.groupby('video_id'):
            group = group.sort_values(by='updated_at')

            prev_position = None
            prev_timestamp = None

            for index, row in group.iterrows():
                current_position = row['current_position']
                current_timestamp = row['updated_at']
                current_title = row['title'] if pd.notna(row['title']) else "Başlık Yok"

                # İlk kayıt mı?
                if prev_timestamp is None:
                    prev_position = current_position
                    prev_timestamp = current_timestamp
                else:
                    # Pozisyon değişmiş mi?
                    if current_position != prev_position:
                        time_diff = current_timestamp - prev_timestamp
                        time_diff_hours = time_diff.total_seconds() / 3600
                        
                        results.append({
                            'video_id': video_id,
                            'title': current_title,
                            'change_timestamp': current_timestamp, 
                            'new_position': current_position,
                            'previous_position': prev_position,
                            'previous_timestamp': prev_timestamp,
                            'hours_since_last_change': round(time_diff_hours, 2)
                        })

                        prev_position = current_position
                        prev_timestamp = current_timestamp
                        
        print("\nThumbnail dosyaları kontrol ediliyor...")
        if os.path.isdir(thumbnail_dir):
            thumbnail_files = {f.split('.')[0] for f in os.listdir(thumbnail_dir) if f.endswith('.jpg')}
            print(f"Klasörde {len(thumbnail_files)} adet .jpg uzantılı dosya bulundu.")
            db_video_ids = set(df['video_id'].unique())
            missing_thumbnails = db_video_ids - thumbnail_files
            extra_thumbnails = thumbnail_files - db_video_ids
            if missing_thumbnails:
                print(f"Uyarı: Veritabanında bulunan ancak thumbnail'i olmayan {len(missing_thumbnails)} video ID var.")
                print(f"Eksik thumbnail ID'leri: {list(missing_thumbnails)[:10]}...") 
            if extra_thumbnails:
                print(f"Bilgi: Thumbnail klasöründe bulunan ancak veritabanında olmayan {len(extra_thumbnails)} ID var.")
                print(f"Fazla thumbnail ID'leri: {list(extra_thumbnails)[:10]}...")
        else:
            print(f"Uyarı: Thumbnail klasörü bulunamadı: {thumbnail_dir}")
            
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig') 
        else:
            print("\nKaydedilecek pozisyon değişikliği bulunamadı.")

    except sqlite3.Error as e:
        print(f"SQLite veritabanı hatası: {e}")
    except FileNotFoundError:
        print(f"Hata: Veritabanı dosyası bulunamadı: {db_path}")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        import traceback
        traceback.print_exc() 
    finally:
        if conn:
            conn.close()
            print("\nVeritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print(f"HATA: Veritabanı dosyası '{DB_FILE}' bulunamadı. Lütfen DB_FILE değişkenini doğru ayarlayın.")
    else:
        analyze_video_positions(DB_FILE, THUMBNAIL_DIR)