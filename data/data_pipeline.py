import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cấu hình đường dẫn
DATA_DIR = 'raw' 
OUTPUT_DIR = 'processed'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_RATIO = 0.7
VALID_RATIO = 0.1

# BƯỚC 1: HÀM ĐỌC VÀ GHÉP DATA 
def load_and_merge_house_data(data_folder, house_name):
    search_pattern = os.path.join(data_folder, f'redd_{house_name}_*.csv')
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        return None

    # Sắp xếp đúng thứ tự số học 
    file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1].replace('.csv', '')))
    
    df_list = []
    for file in file_paths:
        df = pd.read_csv(file, index_col=0)
        df_list.append(df)
        
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"-> Đã gộp {len(file_paths)} files cho {house_name}. Tổng dòng: {len(merged_df)}")
    return merged_df

# BƯỚC 2 & 3: TIỀN XỬ LÝ (Tạo thời gian, Resampling & Normalization) 
COMMON_FEATURES = [
    'dish washer',
    'electric stove',
    'fridge',
    'microwave',
    'washer dryer',
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'lag_1',
    'rolling_mean'
]


def preprocess_pipeline(df):
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='3s')
    
    df_hourly = df.resample('15min').mean().dropna()
    
# --- SỬA: CYCLICAL ENCODING ---
    # Hour (Chu kỳ 24)
    df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    
    # Day of week (Chu kỳ 7)
    df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
    df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
    
    # TIME SERIES FEATURES 
    df_hourly['lag_1'] = df_hourly['main'].shift(1)
    df_hourly['rolling_mean'] = df_hourly['main'].shift(1).rolling(3).mean()
    
    df_hourly = df_hourly.dropna()

    df_hourly = df_hourly.reindex(columns=COMMON_FEATURES + ['main'], fill_value=0)
    
    return df_hourly


# BƯỚC 4: TẠO CHUỖI DỮ LIỆU (Sliding Window cho bài toán Regression) 
def create_sequences(data, window_size=24):
    X, y = [], []

    feature_cols = [col for col in data.columns if col != 'main']

    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size][feature_cols].values)
        y.append(data.iloc[i + window_size]['main'])

    return np.array(X), np.array(y)


# --- THỰC THI CHO 5 NGÔI NHÀ (Mô phỏng đa môi trường)  ---
all_house_data = {}

houses = [1, 2, 3, 4, 6] 

for i in houses:
    house_name = f'house{i}'
    print(f"\nĐang xử lý {house_name}...")
    
    raw_df = load_and_merge_house_data(DATA_DIR, house_name)
    
    if raw_df is not None:
        # 1. Tiền xử lý
        clean_df = preprocess_pipeline(raw_df)    

        # Chia dữ liệu: Train, Valid, Test theo thứ tự thời gian
        total_len = len(clean_df)
        train_end = int(total_len * TRAIN_RATIO)
        valid_end = int(total_len * (TRAIN_RATIO + VALID_RATIO))

        train_df = clean_df.iloc[:train_end].copy()
        valid_df = clean_df.iloc[train_end:valid_end].copy()
        test_df = clean_df.iloc[valid_end:].copy()

        # Xác định các cột cần Scale (tránh scale các cột sin/cos đã chuẩn hóa)
        feature_cols = [col for col in clean_df.columns if col != 'main']
        cols_to_scale = [c for c in feature_cols if '_sin' not in c and '_cos' not in c]

        # 3. scale (FIT TRÊN TRAIN)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        train_df[cols_to_scale] = feature_scaler.fit_transform(train_df[cols_to_scale])
        train_df[['main']] = target_scaler.fit_transform(train_df[['main']])

        # transform val và test
        valid_df[cols_to_scale] = feature_scaler.transform(valid_df[cols_to_scale])
        valid_df[['main']] = target_scaler.transform(valid_df[['main']])

        test_df[cols_to_scale] = feature_scaler.transform(test_df[cols_to_scale])
        test_df[['main']] = target_scaler.transform(test_df[['main']])

        # 4. Tạo chuỗi 
        X_train, y_train = create_sequences(train_df)
        X_valid, y_valid = create_sequences(valid_df)
        X_test, y_test = create_sequences(test_df)
        
        # Lưu trữ để dùng cho Federated Learning sau này
        all_house_data[house_name] = {
            'X_train': X_train, 
            'y_train': y_train,
            'X_valid': X_valid, 
            'y_valid': y_valid,
            'X_test': X_test,
            'y_test': y_test,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }   
              
        print(f"   Train: X={X_train.shape}, y={y_train.shape}")
        print(f"   Valid: X={X_valid.shape}, y={y_valid.shape}")
        print(f"   Test : X={X_test.shape}, y={y_test.shape}")
        
        # Lưu file sạch ra máy để kiểm tra 
        clean_df.to_csv(f"{OUTPUT_DIR}/{house_name}_clean.csv")
        train_df.to_csv(f"{OUTPUT_DIR}/{house_name}_train.csv")
        valid_df.to_csv(f"{OUTPUT_DIR}/{house_name}_valid.csv")
        test_df.to_csv(f"{OUTPUT_DIR}/{house_name}_test.csv")

print("\n--- HOÀN THÀNH TOÀN BỘ DATA PIPELINE ---")