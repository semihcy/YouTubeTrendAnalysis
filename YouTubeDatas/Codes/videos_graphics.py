import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import os
from tqdm.notebook import tqdm 
import warnings


warnings.filterwarnings("ignore")

DB_FILE = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/20250228 - db/youtube_with_ai_TR.db' 
GRAPH_OUTPUT_DIR = '/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/video_trend_graphs/' 

def plot_individual_video_trends(db_path, output_dir):
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
        print("Video pozisyon verileri çekiliyor...")
        df = pd.read_sql_query(query, conn, parse_dates=['updated_at'])
        print(f"Toplam {len(df)} pozisyon kaydı çekildi.")

        if df.empty:
            print("video_stats tablosunda çizilecek veri bulunamadı.")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"Grafikler '{output_dir}' klasörüne kaydedilecek.")

        grouped = df.groupby('video_id')
        total_videos = len(grouped)
        print(f"Toplam {total_videos} benzersiz video için grafik oluşturulacak...")

        for video_id, group in tqdm(grouped, total=total_videos, desc="Grafikler oluşturuluyor"):
            times = group['updated_at']
            positions = group['current_position']
            title = group['title'].iloc[0] if pd.notna(group['title'].iloc[0]) else "Başlık Yok"

            if len(times) < 2:
                continue # Yeterli veri yoksa bu videoyu atla

            try:
                plt.figure(figsize=(12, 6)) # Grafik boyutunu ayarla

                plt.plot(times, positions, marker='.', linestyle='-', markersize=5)

                # Y eksenini ters çevir 
                plt.gca().invert_yaxis()

                plt.xlabel("Zaman")
                plt.ylabel("Sıralama Pozisyonu (Düşük = İyi)")
                plot_title = f"Video Trend Grafiği: {video_id}\n{title[:80]}{'...' if len(title)>80 else ''}" 
                plt.title(plot_title)

                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10)) 
                plt.xticks(rotation=30, ha='right') 

                plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
                plt.tight_layout() 

                safe_video_id = "".join([c if c.isalnum() else "_" for c in video_id])
                filename = f"{safe_video_id}_trend.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)

                plt.close()

            except Exception as e:
                print(f"\nHata: '{video_id}' ID'li video için grafik oluşturulurken hata: {e}")
                plt.close()

        print(f"\nGrafik oluşturma tamamlandı. {total_videos} video işlendi.")

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
        plot_individual_video_trends(DB_FILE, GRAPH_OUTPUT_DIR)