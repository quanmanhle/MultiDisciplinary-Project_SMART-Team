# 📊 Data Pipeline (REDD Dataset)

This module implements the data preprocessing pipeline for the Smart Home Energy Optimization task using the REDD dataset. It prepares time-series data for regression and supports Federated Learning simulation across multiple houses.

## 🎯 Objective

The goal of this pipeline is to prepare time-series data for predicting next-step energy consumption using regression models in a Federated Learning setting.

## 📁 Input Data

Raw data structure:

```
raw/
  redd_house1_0.csv
  redd_house1_1.csv
  redd_house1_2.csv
  ...
  redd_house2_0.csv
```
Files belonging to the same house are automatically merged in correct chronological order.

## ⚙️ Pipeline Steps

### 1. Data Loading & Merging
All CSV files of each house are loaded using pattern matching, sorted numerically to preserve temporal order, and concatenated into a single DataFrame.

### 2. Time Indexing & Resampling
A synthetic timestamp with 3-second frequency is assigned to the dataset. The data is then resampled to 15-minute resolution (15min) using mean aggregation, and missing values are removed.

### 3. Feature Engineering
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
- rolling_mean: moving average (window = 3)  

### 4. Data Cleaning
Rows with missing values are removed. The dataset is aligned to a fixed feature schema including all features and the target variable `main`.

### 5. Train/Test Split
The dataset is split chronologically into 70% training, 10% validation and 20% testing without shuffling.

### 6. Normalization
StandardScaler is applied to normalize the data:
- Fit only on training data  
- Apply to validation and test data  
- Cyclical features (sin, cos) are NOT scaled  

### 7. Sequence Generation
A sliding window approach is used:
- Window size: 24 (i.e., 6 hours with 15-minute intervals)

The model uses the past 24 timesteps (6 hours) to predict the next timestep energy consumption (`main`).

Output:
```
X shape: (num_samples, 24, num_features)
y shape: (num_samples,)
```

### 8. Multi-House Setup (Federated Learning)
Each house is processed independently and stored in a dictionary:

```python
all_house_data = {
    "house1": {
        "X_train": ...,
        "y_train": ...,
        "X_valid": ...,
        "y_valid": ...,
        "X_test": ...,
        "y_test": ...,
        "feature_scaler": ...,
        "target_scaler": ...
    }
}
```

## 💾 Output

Processed data is saved in:

```
processed/
  house1_clean.csv
  house1_train.csv
  house1_valid.csv
  house1_test.csv
```

## 🚀 Key Design Choices

- 15-minute aggregation reduces noise and computation cost  
- Cyclical encoding captures periodic patterns  
- Sliding window converts to supervised learning  
- Per-house data enables Federated Learning  
- Train-only scaling prevents data leakage  

## 🧠 Notes

- Missing appliance columns are filled with 0  
- Assumes consistent column names across files  
- Target variable: `main` (energy consumption)