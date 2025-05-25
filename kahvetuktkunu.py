import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime
from multiprocessing import Pool, cpu_count
import math

def create_kdtree(cafes):
    coords = np.column_stack((cafes['lat'].values, cafes['lng'].values))
    tree = cKDTree(coords)
    return tree, coords

def find_nearby_cafes_batch(args):
    batch, kdtree, cafe_coords, cafe_ids, max_distance, batch_id = args
    matched = []
    batch_coords = np.column_stack((
        batch['latitude'].astype(float).values,
        batch['longitude'].astype(float).values
    ))
    distances, indices = kdtree.query(batch_coords, k=1, distance_upper_bound=max_distance)
    distances_km = distances * 111
    valid_visits = distances_km < max_distance
    for idx, is_valid in enumerate(valid_visits):
        if is_valid:
            ts = batch.iloc[idx]['timestamp']
            if not hasattr(ts, 'date'):
                ts = pd.to_datetime(ts)
            matched.append((
                batch.iloc[idx]['device_aid'],
                cafe_ids[indices[idx]],
                ts,
                batch.index[idx]  # Orijinal Parquet satır numarası
            ))
    return matched

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] İşlem başladı...")

    # 1. Mobility Parquet dosyasını oku
    df = pd.read_parquet("MobilityDataMay2024.parquet")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_full = df.copy()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parquet okundu. Satır: {len(df):,}")

    # 2. Kafe POI'lerini oku
    kafe = pd.read_excel("kafe.xlsx")
    kafe['lat'] = kafe['latitude'].astype(str).str.strip().str.replace("'", "").str.replace(',', '.').astype(float)
    kafe['lng'] = kafe['longitude'].astype(str).str.strip().str.replace("'", "").str.replace(',', '.').astype(float)
    kafe['kafe_id'] = kafe.index.astype(str)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Kafe POI sayısı: {len(kafe):,}")

    # 3. KD-Tree oluştur
    kdtree, cafe_coords = create_kdtree(kafe)
    cafe_ids = kafe['kafe_id'].values

    # 4. Paralel eşleştirme (horizon: 0.05 km)
    max_distance = 0.05
    num_cores = cpu_count()
    batch_size = math.ceil(len(df) / (num_cores * 4))
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    process_args = [
        (batch, kdtree, cafe_coords, cafe_ids, max_distance, i)
        for i, batch in enumerate(batches)
    ]
    with Pool(num_cores) as pool:
        results = pool.map(find_nearby_cafes_batch, process_args)

    # 5. Sonuçları birleştir
    all_matches = [m for batch_matches in results for m in batch_matches]
    matched_df = pd.DataFrame(all_matches, columns=['device_aid', 'kafe_id', 'timestamp', 'orig_index'])

    # 6. GERÇEKÇİ ZİYARET: Aynı kişi, aynı kafede, 6 saatten kısa aralıklarla gelen kayıtlar tek ziyaret
    matched_df['timestamp'] = pd.to_datetime(matched_df['timestamp'])
    matched_df = matched_df.sort_values(['device_aid', 'kafe_id', 'timestamp'])
    matched_df['new_visit'] = (
        matched_df.groupby(['device_aid', 'kafe_id'])['timestamp']
        .diff().dt.total_seconds().fillna(99999) > 6*60*60
    )
    matched_df['visit_group'] = matched_df.groupby(['device_aid', 'kafe_id'])['new_visit'].cumsum()
    realistic_visits = matched_df.groupby(['device_aid', 'kafe_id', 'visit_group']).first().reset_index()
    realistic_counts = realistic_visits.groupby('device_aid').size()
    realistic_counts.name = 'gercekci_kafe_ziyaret_sayisi'

    # 7. Z-SKORU İLE HEAVY USER EŞİĞİ
    mean_visits = realistic_counts.mean()
    std_visits = realistic_counts.std()
    z_threshold = 1.5  # Z-skor eşiği
    z_scores = (realistic_counts - mean_visits) / std_visits
    heavy_users = realistic_counts[z_scores > z_threshold]
    heavy_user_ids = heavy_users.index.tolist()
    print(f"Heavy user z-skor eşiği: {z_threshold}")
    print(f"Ortalama ziyaret: {mean_visits:.2f}, Standart sapma: {std_visits:.2f}")
    print(f"Heavy user sayısı: {len(heavy_users)}")

    # 8. Ziyaret özetlerini kaydet
    heavy_users.to_csv('kahveseverler.csv', header=True, encoding="utf-8-sig")
    print(f"Kahveseverler özet dosyası 'kahveseverler.csv' olarak kaydedildi.")

    # 9. Sadece heavy_users'ın Parquet'teki TÜM hareketlerini Parquet formatında kaydet
    heavy_user_full = df_full[df_full['device_aid'].isin(heavy_user_ids)]
    heavy_user_full.to_parquet('kahveseverler_parquet.parquet', index=False)
    print(f"Kahveseverlerin Parquet'teki TÜM hareketleri 'kahveseverler_parquet.parquet' dosyasına kaydedildi.")

if __name__ == '__main__':
    main()
   