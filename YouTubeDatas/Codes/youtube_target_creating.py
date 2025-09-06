import pandas as pd

POSITION_CHANGES_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/position_changes.csv'
INPUT_FINAL_MERGED_CSV = '/Users/semihcay/PycharmProjects/PythonProject/YouTubeDatas/CSV_Files/multimodal_video_analysis_final_with_channel.csv'
OUTPUT_WITH_TARGET_CSV = 'dataset_for_modeling_with_target.csv'
TARGET_PEAK_POSITION_THRESHOLD = 10


def create_target_from_position_changes(input_merged_csv, position_changes_csv, output_csv, threshold):
    try:
        df_main_merged = pd.read_csv(input_merged_csv)
        print(f"Ana birleştirilmiş veri '{input_merged_csv}' yüklendi ({len(df_main_merged)} kayıt).")
    except FileNotFoundError:
        print(f"HATA: Ana birleştirilmiş CSV dosyası bulunamadı: {input_merged_csv}")
        return
    except Exception as e:
        print(f"HATA: '{input_merged_csv}' dosyası okunurken: {e}")
        return

    try:
        df_pos_changes = pd.read_csv(position_changes_csv)
        print(f"Pozisyon değişimleri dosyası '{position_changes_csv}' yüklendi ({len(df_pos_changes)} kayıt).")
    except FileNotFoundError:
        print(f"HATA: Pozisyon değişimleri CSV dosyası bulunamadı: {position_changes_csv}")
        print("Lütfen bu dosyanın doğru yolda olduğundan emin olun.")
        return
    except Exception as e:
        print(f"HATA: '{position_changes_csv}' dosyası okunurken: {e}")
        return

    if 'video_id' not in df_pos_changes.columns or \
            'new_position' not in df_pos_changes.columns or \
            'previous_position' not in df_pos_changes.columns or \
            'previous_timestamp' not in df_pos_changes.columns:
        print(
            f"HATA: '{position_changes_csv}' dosyasında gerekli ('video_id', 'new_position', 'previous_position', 'previous_timestamp') sütunlar bulunamadı.")
        return

    print(f"UYARI: Hedef değişken için '{position_changes_csv}' kullanılıyor.")
    print("Bu dosya, videonun TÜM pozisyon geçmişini (video_stats'taki gibi) tam olarak yansıtmıyorsa,")
    print("hesaplanan 'peak_position' eksik olabilir. En güvenli yol, `video_stats` tablosundan")
    print("her video_id için `MIN(current_position)` değerini almaktır.")

    #Peak position hesaplama
    min_new_positions_series = df_pos_changes.groupby('video_id')['new_position'].min()
    first_prev_positions_series = df_pos_changes.sort_values(by=['video_id', 'previous_timestamp']) \
        .groupby('video_id') \
        .first()['previous_position']

    df_peak_candidate_new = min_new_positions_series.reset_index(name='peak_candidate_new')
    df_peak_candidate_first_prev = first_prev_positions_series.reset_index(name='peak_candidate_first_prev')

    df_peak_candidates = pd.merge(df_peak_candidate_new, df_peak_candidate_first_prev, on='video_id', how='outer')
    df_peak_candidates['peak_candidate_new'] = df_peak_candidates['peak_candidate_new'].fillna(float('inf'))
    df_peak_candidates['peak_candidate_first_prev'] = df_peak_candidates['peak_candidate_first_prev'].fillna(
        float('inf'))
    df_peak_candidates['peak_position'] = df_peak_candidates[['peak_candidate_new', 'peak_candidate_first_prev']].min(
        axis=1)
    df_peak_candidates.loc[df_peak_candidates['peak_position'] == float('inf'), 'peak_position'] = pd.NA

    df_peak_final = df_peak_candidates[['video_id', 'peak_position']]

    #Trend duration
    df_pos_changes['previous_timestamp'] = pd.to_datetime(df_pos_changes['previous_timestamp'], errors='coerce')
    trend_duration = df_pos_changes.groupby('video_id')['previous_timestamp'].nunique().reset_index(
        name='trend_duration')

    #Time to peak
    def calc_time_to_peak(group):
        if group['previous_timestamp'].isnull().all():
            return pd.NA
        first_day = group['previous_timestamp'].min()
        best_pos = group['new_position'].min()
        best_day = group.loc[group['new_position'] == best_pos, 'previous_timestamp'].min()
        return (best_day - first_day).days if pd.notna(first_day) and pd.notna(best_day) else pd.NA

    time_to_peak = df_pos_changes.groupby('video_id').apply(calc_time_to_peak).reset_index(name='time_to_peak')

    df_with_target = pd.merge(df_main_merged, df_peak_final, on='video_id', how='left')
    df_with_target = pd.merge(df_with_target, trend_duration, on='video_id', how='left')
    df_with_target = pd.merge(df_with_target, time_to_peak, on='video_id', how='left')

    print(f"Ek hedefler ana veri setine eklendi. Kayıt sayısı: {len(df_with_target)}")

    #Binary target
    target_col_name = f'did_it_reach_the_top{threshold}'
    df_with_target[target_col_name] = (df_with_target['peak_position'] <= threshold)
    df_with_target[target_col_name] = df_with_target[target_col_name].astype('Int64')

    print(f"\nOluşturulan hedef değişken ('{target_col_name}') dağılımı:")
    print(df_with_target[target_col_name].value_counts(dropna=False))

    nan_target_count = df_with_target[target_col_name].isnull().sum()
    if nan_target_count > 0:
        print(f"Uyarı: {nan_target_count} video için '{target_col_name}' (hedef değişken) NaN olarak kaldı.")
        print("Bu, bu videolar için pozisyon bilgisi bulunamadığı anlamına gelir.")

    try:
        df_with_target.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Hedef değişken eklenmiş, modellemeye hazır veri '{output_csv}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"HATA: Sonuç dosyası '{output_csv}' kaydedilemedi: {e}")


if __name__ == '__main__':
    create_target_from_position_changes(
        INPUT_FINAL_MERGED_CSV,
        POSITION_CHANGES_CSV,
        OUTPUT_WITH_TARGET_CSV,
        TARGET_PEAK_POSITION_THRESHOLD
    )
