import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
REDD_DIR    = 'raw/REDD dataset'
UKDALE_DIR  = 'raw/UK-DALE dataset'
OUTPUT_DIR  = 'processed'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_RATIO = 0.7
VALID_RATIO = 0.1

# ============================================================
# FEATURES CHUNG 
# ============================================================
COMMON_FEATURES = [
    'dish washer',
    'electric stove',
    'fridge',
    'microwave',
    'washer dryer',
    'hour_sin', 'hour_cos',
    'day_sin',  'day_cos',
    'lag_1',
    'rolling_mean'
]

# ============================================================
# MAPPING APPLIANCE UK-DALE: tên feature → channel number
# channel_1 luôn là aggregate (main)
# None → không có thiết bị đó → fill_value=0
# ============================================================
UKDALE_APPLIANCE_MAP = {
    'house_1': {
        'dish washer':    6,    # dishwasher
        'electric stove': None, # không có
        'fridge':         12,   # fridge
        'microwave':      13,   # microwave
        'washer dryer':   5,    # washing_machine
    },
    'house_2': {
        'dish washer':    13,   # dish_washer
        'electric stove': 19,   # cooker
        'fridge':         14,   # fridge
        'microwave':      15,   # microwave
        'washer dryer':   12,   # washing_machine
    },
    'house_3': {
        'dish washer':    None, # không có
        'electric stove': None, # không có
        'fridge':         None, # không có
        'microwave':      None, # không có
        'washer dryer':   None, # không có
        # house_3 chỉ có kettle, electric_heater, laptop, projector
        # → appliance features fill 0, model học từ main + time + lag
    },
    'house_4': {
        'dish washer':    None, # không có
        'electric stove': None, # gas_boiler không đo điện
        'fridge':         5,    # freezer ~ fridge 
        'microwave':      None, # gộp trong ch_6, tránh double-count
        'washer dryer':   6,    # washing_machine_microwave_breadmaker (mixed channel)
    },
    'house_5': {
        'dish washer':    22,   # dishwasher
        'electric stove': 21,   # electric_hob 
        'fridge':         19,   # fridge_freezer
        'microwave':      23,   # microwave
        'washer dryer':   24,   # washer_dryer
    },
}


# ============================================================
# SHARED UTILITIES
# ============================================================
def create_sequences(data, window_size=24):
    """
    Sliding window: lấy 24 bước (= 6 tiếng) để dự đoán bước tiếp theo.
    Input : DataFrame đã scale, cột cuối là 'main'
    Output: X shape (N, 24, n_features), y shape (N,)
    """
    X, y = [], []
    feature_cols = [col for col in data.columns if col != 'main']
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size][feature_cols].values)
        y.append(data.iloc[i + window_size]['main'])
    return np.array(X), np.array(y)


def scale_and_split(clean_df):
    """
    Chia train/valid/test theo thứ tự thời gian,
    fit StandardScaler trên train rồi transform valid/test.
    Trả về dict chứa arrays và scalers.
    """
    total_len = len(clean_df)
    train_end = int(total_len * TRAIN_RATIO)
    valid_end = int(total_len * (TRAIN_RATIO + VALID_RATIO))

    train_df = clean_df.iloc[:train_end].copy()
    valid_df = clean_df.iloc[train_end:valid_end].copy()
    test_df  = clean_df.iloc[valid_end:].copy()

    # Không scale các cột sin/cos (đã trong [-1, 1])
    feature_cols  = [col for col in clean_df.columns if col != 'main']
    cols_to_scale = [c for c in feature_cols if '_sin' not in c and '_cos' not in c]

    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()

    # Fit chỉ trên train → tránh data leakage
    train_df[cols_to_scale] = feature_scaler.fit_transform(train_df[cols_to_scale])
    train_df[['main']]      = target_scaler.fit_transform(train_df[['main']])

    valid_df[cols_to_scale] = feature_scaler.transform(valid_df[cols_to_scale])
    valid_df[['main']]      = target_scaler.transform(valid_df[['main']])

    test_df[cols_to_scale]  = feature_scaler.transform(test_df[cols_to_scale])
    test_df[['main']]       = target_scaler.transform(test_df[['main']])

    X_train, y_train = create_sequences(train_df)
    X_valid, y_valid = create_sequences(valid_df)
    X_test,  y_test  = create_sequences(test_df)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_valid': X_valid, 'y_valid': y_valid,
        'X_test':  X_test,  'y_test':  y_test,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler
    }


def add_time_features(df):
    """
    Thêm cyclical encoding + lag features (dùng chung cho cả REDD và UK-DALE).
    """
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin']  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos']  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['lag_1']        = df['main'].shift(1)
    df['rolling_mean'] = df['main'].shift(1).rolling(3).mean()
    df = df.dropna()
    df = df.reindex(columns=COMMON_FEATURES + ['main'], fill_value=0)
    return df


def save_csv(clean_df, train_df, valid_df, test_df, name):
    """Lưu các split ra CSV để kiểm tra."""
    clean_df.to_csv(f"{OUTPUT_DIR}/{name}_clean.csv")
    train_df.to_csv(f"{OUTPUT_DIR}/{name}_train.csv")
    valid_df.to_csv(f"{OUTPUT_DIR}/{name}_valid.csv")
    test_df.to_csv(f"{OUTPUT_DIR}/{name}_test.csv")


# ============================================================
# PHẦN 1: REDD PIPELINE (house1–5, bỏ house5 do thiếu data)
# ============================================================
def load_and_merge_redd(data_folder, house_name):
    """Đọc và ghép các file CSV của 1 house REDD."""
    search_pattern = os.path.join(data_folder, f'redd_{house_name}_*.csv')
    file_paths = glob.glob(search_pattern)
    if not file_paths:
        return None
    file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1].replace('.csv', '')))
    df_list = [pd.read_csv(f, index_col=0) for f in file_paths]
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"   -> Đã gộp {len(file_paths)} files. Tổng dòng: {len(merged_df)}")
    return merged_df


def preprocess_redd(df):
    """
    REDD không có timestamp thật → tạo fake index tần suất 3s.
    Sau đó resample về 15min và thêm time features.
    """
    df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='3s')
    df_15min = df.resample('15min').mean().dropna()
    return add_time_features(df_15min)


def process_redd(all_house_data):
    """Xử lý 5 house REDD → house1..house5."""
    # house5 trong REDD thực tế là house6 (house5 thiếu data)
    house_mapping = {
        'house1': 'house1',
        'house2': 'house2',
        'house3': 'house3',
        'house4': 'house4',
        'house5': 'house6',
    }

    for display_name, real_house in house_mapping.items():
        print(f"\nĐang xử lý REDD {real_house} → {display_name}...")

        raw_df = load_and_merge_redd(REDD_DIR, real_house)
        if raw_df is None:
            print(f"   [SKIP] Không tìm thấy file cho {real_house}")
            continue

        clean_df = preprocess_redd(raw_df)

        result = scale_and_split(clean_df)
        all_house_data[display_name] = result

        print(f"   Train: X={result['X_train'].shape}, y={result['y_train'].shape}")
        print(f"   Valid: X={result['X_valid'].shape}, y={result['y_valid'].shape}")
        print(f"   Test : X={result['X_test'].shape},  y={result['y_test'].shape}")

        # Lưu CSV (tái tạo split từ clean_df để save)
        total_len = len(clean_df)
        train_end = int(total_len * TRAIN_RATIO)
        valid_end = int(total_len * (TRAIN_RATIO + VALID_RATIO))
        save_csv(
            clean_df,
            clean_df.iloc[:train_end],
            clean_df.iloc[train_end:valid_end],
            clean_df.iloc[valid_end:],
            display_name
        )

    return all_house_data


# ============================================================
# PHẦN 2: UK-DALE PIPELINE (house_1–5 → house6..house10)
# ============================================================
def load_dat_channel(house_folder, channel_num):
    """
    Đọc file channel_X.dat.
    Format mỗi dòng: <unix_timestamp>  <watt>
    Trả về DataFrame 1 cột 'watt' với DatetimeIndex (no timezone).
    """
    file_path = os.path.join(house_folder, f'channel_{channel_num}.dat')
    if not os.path.exists(file_path):
        print(f"   [WARN] Không tìm thấy: {file_path}")
        return None

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        names=['timestamp', 'watt']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.set_index('timestamp').sort_index()
    df.index = df.index.tz_convert(None)  # bỏ timezone → đồng nhất với REDD
    return df[['watt']]


def load_ukdale_house(house_name, appliance_map):
    """
    Đọc channel_1 (main) + appliance channels của 1 house UK-DALE.
    Resample về 15min, ghép thành DataFrame cùng cấu trúc với REDD.
    """
    house_folder = os.path.join(UKDALE_DIR, house_name)

    main_df = load_dat_channel(house_folder, 1)
    if main_df is None:
        print(f"   [ERROR] Không đọc được main cho {house_name}")
        return None

    main_15min = main_df.resample('15min').mean()
    main_15min.columns = ['main']

    appliance_dfs = {}
    for appliance_name, channel_num in appliance_map.items():
        if channel_num is None:
            appliance_dfs[appliance_name] = pd.Series(
                np.nan, index=main_15min.index, name=appliance_name
            )
        else:
            ch_df = load_dat_channel(house_folder, channel_num)
            if ch_df is not None:
                resampled = ch_df['watt'].resample('15min').mean()
                resampled.name = appliance_name
                appliance_dfs[appliance_name] = resampled
            else:
                appliance_dfs[appliance_name] = pd.Series(
                    np.nan, index=main_15min.index, name=appliance_name
                )

    merged = pd.concat([main_15min] + list(appliance_dfs.values()), axis=1)
    merged = merged.dropna(subset=['main'])
    merged = merged.fillna(0)

    print(f"   -> {house_name}: {len(merged)} rows sau resample 15min")
    return merged


def process_ukdale(all_house_data, start_index=6):
    """Xử lý 5 house UK-DALE → house6..house10."""
    ukdale_houses = ['house_1', 'house_2', 'house_3', 'house_4', 'house_5']

    for idx, house_name in enumerate(ukdale_houses):
        client_name = f'house{start_index + idx}'
        print(f"\nĐang xử lý UK-DALE {house_name} → {client_name}...")

        appliance_map = UKDALE_APPLIANCE_MAP.get(house_name, {})

        raw_df = load_ukdale_house(house_name, appliance_map)
        if raw_df is None:
            print(f"   [SKIP] Bỏ qua {house_name}")
            continue

        clean_df = add_time_features(raw_df)

        if len(clean_df) < 100:
            print(f"   [SKIP] Quá ít dữ liệu ({len(clean_df)} rows)")
            continue

        result = scale_and_split(clean_df)
        all_house_data[client_name] = result

        print(f"   Train: X={result['X_train'].shape}, y={result['y_train'].shape}")
        print(f"   Valid: X={result['X_valid'].shape}, y={result['y_valid'].shape}")
        print(f"   Test : X={result['X_test'].shape},  y={result['y_test'].shape}")

        total_len = len(clean_df)
        train_end = int(total_len * TRAIN_RATIO)
        valid_end = int(total_len * (TRAIN_RATIO + VALID_RATIO))
        save_csv(
            clean_df,
            clean_df.iloc[:train_end],
            clean_df.iloc[train_end:valid_end],
            clean_df.iloc[valid_end:],
            client_name
        )

    return all_house_data


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    all_house_data = {}

    print("=" * 55)
    print("PHẦN 1: XỬ LÝ REDD (house1–house5)")
    print("=" * 55)
    all_house_data = process_redd(all_house_data)

    print("\n" + "=" * 55)
    print("PHẦN 2: XỬ LÝ UK-DALE (house6–house10)")
    print("=" * 55)
    all_house_data = process_ukdale(all_house_data, start_index=6)

    print("\n" + "=" * 55)
    print("HOÀN THÀNH TOÀN BỘ DATA PIPELINE")
    print("=" * 55)
    print(f"Tổng số FL clients : {len(all_house_data)}")
    print(f"Danh sách          : {list(all_house_data.keys())}")
    for name, data in all_house_data.items():
        print(f"  {name}: train={data['X_train'].shape}, "
              f"valid={data['X_valid'].shape}, "
              f"test={data['X_test'].shape}")