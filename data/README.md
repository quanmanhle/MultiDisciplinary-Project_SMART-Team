# 📊 Data Pipeline (REDD & UK-DALE Datasets)

This module implements the data preprocessing pipeline for the Smart Home Energy Optimization task. It integrates both the REDD (US) and UK-DALE (UK) datasets to simulate a highly realistic, Non-IID Federated Learning environment with a total of 10 distinct houses (clients).

## 🎯 Objective

The goal of this pipeline is to prepare and align heterogeneous time-series data from two different public datasets. It processes them into a unified feature space to predict next-step energy consumption in a Federated Learning setting.

## 📁 Input Data

Raw data structure:

```
raw/
  ├── REDD dataset/
  │     ├── redd_house1_1.csv
  │     ├── redd_house1_2.csv
  │     └── ...
  └── UK-DALE dataset/
        ├── house_1/
        │     ├── channel_1.dat (main)
        │     ├── channel_6.dat
        │     └── ...
        └── house_2/
```


## ⚙️ Pipeline Steps

### 1. Data Loading & Merging
- **REDD Dataset:** All CSV files for a specific house are matched using patterns, sorted numerically to preserve temporal order, and concatenated.
- **UK-DALE Dataset:** Data is read from .dat files. channel_1.dat is strictly treated as the aggregate/main power. Appliance channels are loaded based on a predefined mapping dictionary.

### 2. Time Indexing & Resampling
- **REDD:** A synthetic timestamp with a 3-second frequency is assigned to the dataset.
- **UK-DALE:** Actual UNIX timestamps are converted to timezone-naive Datetime objects.
- **Both:** The data is resampled to a unified 15-minute resolution (15min) using mean aggregation.

### 3. Feature Mapping & Engineering
To ensure all 10 clients share the exact same input architecture for Federated Learning, appliances from UK-DALE are mapped to REDD's standard names using UKDALE_APPLIANCE_MAP. Missing appliances in any house are automatically filled with 0.

**Appliance Features:**
- dish washer  
- electric stove  
- fridge  
- microwave  
- washer dryer  

**Cyclical Time Encoding:**
- hour_sin, hour_cos  
- day_sin, day_cos  

**Time-Series Features:**
- lag_1: previous timestep consumption  
- rolling_mean: moving average of the previous 3 timesteps

### 4. Data Cleaning
Rows with missing values in the target variable are removed. The dataset is reindexed to enforce a strict schema containing the common features plus the target variable `main`.

### 5. Train/Test Split
The dataset is split chronologically without shuffling to preserve time-series integrity:
- 70% Training
- 10% Validation
- 20% Testing

### 6. Normalization
`StandardScaler` is applied to normalize the power values:
- Fit only on the training set to prevent data leakage.
- Transformed on validation and test sets.  
- Cyclical time features (sin, cos) are explicitly excluded from scaling as they are already bounded [-1, 1]. 

### 7. Sequence Generation (Sliding Window)
A sliding window approach converts the continuous time-series into supervised learning samples:
- Window size: 24 (representing 6 hours of 15-minute intervals).
- The model learns from the past 24 timesteps of all features to predict the energy consumption (`main`) of the next timestep.

Output shapes per client:
```
X shape: (num_samples, 24, 11)  # 11 features
y shape: (num_samples,)
```

### 8. Multi-House Setup (10 FL Clients)
The pipeline generates data for 10 independent clients. **House 1 to 5** belong to REDD, and **House 6 to 10** belong to UK-DALE. The processed arrays and scalers are stored in a dictionary:

```python
all_house_data = {
    "house1": { ... }, # REDD house 1
    ...
    "house6": {        # UK-DALE house_1
        "X_train": ...,
        "y_train": ...,
        "X_valid": ...,
        "y_valid": ...,
        "X_test": ...,
        "y_test": ...,
        "feature_scaler": ...,
        "target_scaler": ...
    },
    ...
    "house10": { ... } # UK-DALE house_5
}
```

## 💾 Output

Processed data is saved in:

```
processed/
  ├── house1_clean.csv
  ├── house1_train.csv
  ├── house1_valid.csv
  ├── house1_test.csv
  ├── ...
  └── house10_test.csv
```

## 🚀 Key Design Choices

- **Cross-Dataset Integration:** Merging REDD and UK-DALE creates a robust Non-IID scenario (varying data volumes, different missing appliances) crucial for realistic Federated Learning evaluation.
- **Unified Feature Alignment:** The reindex(fill_value=0) technique guarantees all local models have the exact same input dimension (24, 11), despite hardware differences between homes.
- **Strict Chronological Splits:** Prevents look-ahead bias in time-series forecasting.
- **Train-only Scaling:** Ensures zero data leakage between splits.