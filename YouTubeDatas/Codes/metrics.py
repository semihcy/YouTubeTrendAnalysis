import pandas as pd
import numpy as np 

ANALYSIS_CSV_FILE = 'thumbnail_trend_analysis.csv'
CLUSTER_COLUMN = 'thumbnail_cluster'
METRIC_COLUMNS = [
    'start_position',
    'end_position',
    'net_change',
    'avg_hourly_change',
    'total_hours',
    'avg_rise_speed', 
    'avg_fall_speed'
]

DECIMAL_PLACES = 2
LATEX_COLUMN_MAPPING = {
    'Video Sayısı': 'Video Sayısı',
    'start_position': 'Ort. Başl. Poz.',
    'end_position': 'Ort. Bitiş Poz.',
    'net_change': 'Ort. Net Değişim',
    'avg_hourly_change': 'Ort. Saatlik Değişim',
    'avg_rise_speed': 'Ort. Yüks. Hızı',
    'avg_fall_speed': 'Ort. Düşüş Hızı',
    'total_hours': 'Ort. Top. Süre (Saat)'
}

def calculate_cluster_stats(csv_filepath, cluster_col, metric_cols, round_digits, latex_mapping):
    try:
        df = pd.read_csv(csv_filepath)
        print(f"'{csv_filepath}' dosyası başarıyla yüklendi.")

        if cluster_col not in df.columns:
            print(f"HATA: Küme sütunu '{cluster_col}' dosyada bulunamadı.")
            return None

        valid_metric_cols = [col for col in metric_cols if col in df.columns]
        missing_metric_cols = [col for col in metric_cols if col not in df.columns]

        if missing_metric_cols:
            print(f"Uyarı: Şu metrik sütunları bulunamadı ve atlanacak: {', '.join(missing_metric_cols)}")

        if not valid_metric_cols:
            print("Hata: Ortalaması alınacak geçerli metrik sütunu bulunamadı.")
            return None

        grouped = df.groupby(cluster_col)
        agg_dict = {'video_id': 'count'} 
        for col in valid_metric_cols:
            agg_dict[col] = 'mean'

        cluster_summary = grouped.agg(agg_dict)

        cluster_summary = cluster_summary.rename(columns={'video_id': 'Video Sayısı'})

        # Sayısal değerleri yuvarla (Video Sayısı hariç)
        cols_to_round = [col for col in cluster_summary.columns if col != 'Video Sayısı']
        cluster_summary[cols_to_round] = cluster_summary[cols_to_round].round(round_digits)

        print("\n--- Küme Bazlı Ortalama Trend Metrikleri ---")
        print(cluster_summary)
        print("--------------------------------------------")

        # LaTeX tablosu için formatlanmış çıktı
        print("\n--- LaTeX Tablosu İçin Satırlar (Yaklaşık) ---")
        latex_ordered_cols = list(latex_mapping.keys())
        available_latex_cols = [col for col in latex_ordered_cols if col in cluster_summary.columns or col in latex_mapping]

        header = "Küme ID & " + " & ".join([latex_mapping[col] for col in available_latex_cols]) + " \\\\"
        print(header)
        print("\\hline")

        for index, row in cluster_summary.iterrows():
            row_values = [str(index)] 
            for col_key in available_latex_cols:
                if col_key == 'Video Sayısı':
                    row_values.append(str(int(row['Video Sayısı']))) 
                elif col_key in row.index: 
                    value = row[col_key]
                    if pd.isna(value):
                        row_values.append("N/A") 
                    else:
                         try:
                           row_values.append(f"{value:.{round_digits}f}")
                         except (TypeError, ValueError):
                           row_values.append(str(value))
                else:
                     row_values.append("?")

            print(" & ".join(row_values) + " \\\\")
        print("\\hline")
        try:
            overall_means = df[valid_metric_cols].mean().round(round_digits)
            total_videos = len(df)
            overall_row = ["\\textbf{Genel Ort.}"]
            for col_key in available_latex_cols:
                 if col_key == 'Video Sayısı':
                     overall_row.append(f"\\textbf{{{total_videos}}}")
                 elif col_key in overall_means.index:
                     value = overall_means[col_key]
                     if pd.isna(value):
                         overall_row.append("\\textbf{N/A}")
                     else:
                          try:
                            overall_row.append(f"\\textbf{{{value:.{round_digits}f}}}")
                          except (TypeError, ValueError):
                            overall_row.append(f"\\textbf{{{value}}}") 
                 else:
                      overall_row.append("\\textbf{?}")
            print(" & ".join(overall_row) + " \\\\")
            print("\\hline")
        except Exception as e:
             print(f"Genel ortalama hesaplanırken hata: {e}")


        print("------------------------------------------")


        return cluster_summary

    except FileNotFoundError:
        print(f"HATA: CSV dosyası bulunamadı: {csv_filepath}")
        return None
    except KeyError as e:
        print(f"HATA: Sütun bulunamadı - {e}. CSV dosyasında '{cluster_col}' sütununun olduğundan emin olun.")
        return None
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    calculate_cluster_stats(ANALYSIS_CSV_FILE, CLUSTER_COLUMN, METRIC_COLUMNS, DECIMAL_PLACES, LATEX_COLUMN_MAPPING)