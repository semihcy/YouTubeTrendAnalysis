import sqlite3
import pandas as pd
import requests
import os

conn = sqlite3.connect('/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/20250228 - db/youtube_with_ai_TR.db')
videos_df = pd.read_sql_query("SELECT video_id, thumbnail_maxres FROM videos", conn)

conn.close()

thumbnail_folder = "/Users/semihcay/Desktop/Bitirme Ödevi/YouTubeDatas/SOA-2024-13927/thumbnails_tr/"
os.makedirs(thumbnail_folder, exist_ok=True)

for idx, row in videos_df.iterrows():
    video_id = row['video_id']
    thumbnail_url = row['thumbnail_maxres']
    
    if pd.isna(thumbnail_url) or thumbnail_url.strip() == "":
        continue  
    
    try:
        response = requests.get(thumbnail_url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(thumbnail_folder, f"{video_id}.jpg"), "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download thumbnail for video_id: {video_id}")
    except Exception as e:
        print(f"Error downloading thumbnail for video_id: {video_id}, error: {e}")









