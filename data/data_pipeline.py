import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cấu hình đường dẫn
DATA_DIR = 'raw' 
OUTPUT_DIR = 'processed'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
    'hour',
    'day',
    'lag_1',
    'rolling_mean'
]


def preprocess_pipeline(df):
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='3s')
    
    df_hourly = df.resample('1h').mean().dropna()
    
    # TIME FEATURES
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['day'] = df_hourly.index.dayofweek
    
    # TIME SERIES FEATURES 
    df_hourly['lag_1'] = df_hourly['main'].shift(1)
    df_hourly['rolling_mean'] = df_hourly['main'].shift(1).rolling(3).mean()
    
    df_hourly = df_hourly.dropna()

    df_hourly = df_hourly.reindex(columns=COMMON_FEATURES + ['main'], fill_value=0)
    
    target = df_hourly['main']
    features = df_hourly.drop(columns=['main'])

    # 1. Scaler cho Features
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)

    # 2. Scaler ĐỘC LẬP cho Target (cần reshape vì target là 1D array)
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # Ghép lại vào DataFrame
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=df_hourly.index)
    df_scaled['main'] = scaled_target # Target đã được scale [0, 1]

    # Trả về cả 2 scaler để dùng cho lúc test
    return df_scaled, feature_scaler, target_scaler


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
        # Tiền xử lý
        clean_df, feature_scaler, target_scaler = preprocess_pipeline(raw_df)        
        # Tạo chuỗi Input/Output cho mô hình
        X, y = create_sequences(clean_df)
        
        # Lưu trữ để dùng cho Federated Learning sau này
        all_house_data[house_name] = {
                    'X': X, 
                    'y': y,
                    'feature_scaler': feature_scaler,
                    'target_scaler': target_scaler
                }  
              
        print(f"   Kích thước X: {X.shape} | Kích thước y: {y.shape}")
        
        # Lưu file sạch ra máy để kiểm tra 
        clean_df.to_csv(f"{OUTPUT_DIR}/{house_name}_hourly_clean.csv")

print("\n--- HOÀN THÀNH TOÀN BỘ DATA PIPELINE ---")