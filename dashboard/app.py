import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import glob

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Smart Home Energy Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Smart Home Energy Optimization")
st.markdown("Federated Learning based Smart Home Energy Monitoring")
st.divider()

# ===== HÀM TIỆN ÍCH =====
def progress_circle(value, title, height=150):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#e6f3ff"},
                {'range': [50, 100], 'color': "#b3d9ff"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def generate_energy_data(days=1):
    hours = 24 * days
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [base + timedelta(hours=i) for i in range(hours)]
    hourly_pattern = 5 + 3 * np.sin(np.linspace(0, 2*np.pi, hours))
    consumption = hourly_pattern + np.random.normal(0, 0.5, hours)
    consumption = np.maximum(consumption, 0)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'consumption_kwh': consumption
    })
    return df

# ===== XÁC ĐỊNH ĐƯỜNG DẪN =====
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir)  
MODELS_RESULTS_DIR = os.path.join(BASE_DIR, "models", "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

def find_file(filename, start_dir=BASE_DIR):
    if os.path.exists(os.path.join(MODELS_RESULTS_DIR, filename)):
        return os.path.join(MODELS_RESULTS_DIR, filename)
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_dir(dirname, start_dir=BASE_DIR):
    if os.path.exists(os.path.join(MODELS_RESULTS_DIR, dirname)):
        return os.path.join(MODELS_RESULTS_DIR, dirname)
    for root, dirs, files in os.walk(start_dir):
        if dirname in dirs:
            return os.path.join(root, dirname)
    return None


metrics_file = find_file("local_metrics.csv") or find_file("localMetrics.csv")
plots_dir = find_dir("plots")
predictions_dir = find_dir("predictions")
fl_log_path = find_file("fl_round_logs.csv")

# ===== ĐỌC DỮ LIỆU =====
def load_house_data(house_name):
    file_path = os.path.join(DATA_PATH, f"{house_name}_hourly_clean.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return None

def load_all_houses():
    houses = ["house1", "house2", "house3", "house4", "house6"]
    data = {}
    for h in houses:
        df = load_house_data(h)
        if df is not None:
            data[h] = df
    return data

def load_fl_logs():
    if fl_log_path and os.path.exists(fl_log_path):
        return pd.read_csv(fl_log_path)
    return None

def load_local_metrics():
    if metrics_file and os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        # Đảm bảo tên cột house là chuẩn
        if 'house' not in df.columns:
            if 'House' in df.columns:
                df = df.rename(columns={'House': 'house'})
            else:
                # Thử tìm cột đầu tiên
                first_col = df.columns[0]
                if first_col.lower() == 'house':
                    df = df.rename(columns={first_col: 'house'})
        return df
    return None

def load_predictions(house_name):
    if predictions_dir:
        file_path = os.path.join(predictions_dir, f"{house_name}_predictions.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            return df
    return None

houses_data = load_all_houses()
fl_logs = load_fl_logs()
local_metrics = load_local_metrics()

# ===== SIDEBAR: HIỂN THỊ BIỂU ĐỒ =====
with st.sidebar:
    st.header("📈 Biểu đồ dự đoán")
    st.markdown("Chọn hộ và mô hình để xem biểu đồ chi tiết")

    if houses_data:
        house_list = list(houses_data.keys())
    else:
        house_list = ["house1", "house2", "house3", "house4", "house6"]

    selected_house = st.selectbox("Chọn hộ", house_list, format_func=lambda x: x.capitalize())
    model_type = st.selectbox(
        "Chọn mô hình",
        options=["linear_regression", "naive_baseline"],
        format_func=lambda x: "Linear Regression" if x == "linear_regression" else "Naive Baseline"
    )

    # Tìm ảnh trong plots_dir
    img_path = None
    if plots_dir:
        possible_ext = ['.png']
        for ext in possible_ext:
            candidate = os.path.join(plots_dir, f"{selected_house}_{model_type}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
    if img_path:
        st.image(img_path, caption=f"{selected_house.capitalize()} - {model_type.replace('_', ' ').title()}", use_container_width=True)
    else:
        st.info(f"Chưa có biểu đồ cho {selected_house} - {model_type}")

# ===== HÀNG 1: TỔNG QUAN HỆ THỐNG =====
st.header("📊 Tổng quan hệ thống")
row1_col1, row1_col2 = st.columns([1, 2])

with row1_col1:
    st.subheader("Mức độ tham gia của các hộ")
    house_order = ["house1", "house2", "house3", "house4", "house6"]
    if houses_data:
        totals = {house: df['main'].sum() for house, df in houses_data.items()}
        total_all = sum(totals.values())
        percentages = {house: (totals[house]/total_all)*100 for house in house_order}
    else:
        percentages = {
            "house1": 22, "house2": 16, "house3": 28,
            "house4": 19, "house6": 15
        }
    col_a, col_b = st.columns(2)
    with col_a:
        for house in house_order[:3]:
            st.plotly_chart(progress_circle(percentages[house], house.capitalize(), height=120), width='stretch')
    with col_b:
        for house in house_order[3:]:
            st.plotly_chart(progress_circle(percentages[house], house.capitalize(), height=120), width='stretch')

with row1_col2:
    st.subheader("Dự báo tiêu thụ hôm nay (kWh)")
    if 'house1' in houses_data:
        df_house = houses_data['house1']
        last_24h = df_house.iloc[-24:].copy()
        last_24h['hour'] = last_24h.index.hour
        hourly = last_24h.groupby('hour')['main'].mean().reset_index()
        st.bar_chart(hourly.set_index('hour'))
        total = last_24h['main'].sum()
        peak_row = hourly.loc[hourly['main'].idxmax()]
        peak_hour = int(peak_row['hour'])
        peak_val = peak_row['main']
        avg = last_24h['main'].mean()
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Tổng dự báo", f"{total:.2f}", "-5%")
        with col_stat2:
            st.metric("Giờ cao điểm", f"{peak_hour}h", f"{peak_val:.2f} kWh")
        with col_stat3:
            st.metric("Trung bình", f"{avg:.2f}", "+2%")
    else:
        df_energy = generate_energy_data(days=1)
        df_energy['hour'] = df_energy['timestamp'].dt.hour
        hourly = df_energy.groupby('hour')['consumption_kwh'].mean().reset_index()
        st.bar_chart(hourly.set_index('hour'))
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Tổng dự báo", f"{df_energy['consumption_kwh'].sum():.1f} kWh", "-5%")
        with col_stat2:
            peak = hourly.loc[hourly['consumption_kwh'].idxmax()]
            st.metric("Giờ cao điểm", f"{int(peak['hour'])}h", f"{peak['consumption_kwh']:.1f} kWh")
        with col_stat3:
            st.metric("Trung bình", f"{df_energy['consumption_kwh'].mean():.1f} kWh", "+2%")

st.divider()

st.header("📈 Trạng thái các client & so sánh mô hình")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Thông tin các hộ tham gia")
    if houses_data:
        clients_list = []
        for house_name, df in houses_data.items():
            total_consumption = df['main'].sum()
            avg_consumption = df['main'].mean()
            # Lấy R² từ local_metrics
            r2_str = "..."
            if local_metrics is not None:
                mask = (local_metrics['house'] == house_name) & (local_metrics['model'] == 'linear_regression')
                if mask.any():
                    r2_val = local_metrics.loc[mask, 'r2'].values[0]
                    r2_str = f"{r2_val:.3f}"
            # Lấy FL logs
            last_round = "..."
            last_loss = "..."
            if fl_logs is not None and not fl_logs.empty:
                last_round = fl_logs['round'].iloc[-1]
                last_loss = fl_logs['loss'].iloc[-1] if 'loss' in fl_logs else "..."
            devices = np.random.randint(3, 8)
            clients_list.append({
                "Hộ": house_name,
                "Tổng tiêu thụ (scaled)": f"{total_consumption:.2f}",
                "Trung bình (scaled)": f"{avg_consumption:.2f}",
                "Vòng FL": last_round,
                "Loss": f"{last_loss:.4f}" if isinstance(last_loss, float) else last_loss,
                "R² (local)": r2_str,
                "Thiết bị": devices
            })
        df_clients = pd.DataFrame(clients_list)
        st.dataframe(df_clients, use_container_width=True, hide_index=True)
        st.caption("🔧 *R² (local) từ linear regression; các cột FL là placeholder cho đến khi có log.*")
    else:
        st.info("Không có dữ liệu hộ. Kiểm tra thư mục data/processed.")

with row2_col2:
    st.subheader("So sánh hiệu năng các mô hình")
    if local_metrics is not None:
        naive = local_metrics[local_metrics['model'] == 'naive_baseline']
        lr = local_metrics[local_metrics['model'] == 'linear_regression']
        naive_avg = naive[['mae','rmse','r2']].mean()
        lr_avg = lr[['mae','rmse','r2']].mean()
        model_r2 = {
            "Naive (baseline)": naive_avg['r2'],
            "Local (linear)": lr_avg['r2'],
            "Federated (FL)": 0.84
        }
        df_models = pd.DataFrame.from_dict(model_r2, orient='index', columns=['R²'])
        st.bar_chart(df_models)
        col_err1, col_err2, col_err3 = st.columns(3)
        with col_err1:
            st.metric("MAE - Naive", f"{naive_avg['mae']:.3f}", "±0.05")
        with col_err2:
            st.metric("MAE - Local", f"{lr_avg['mae']:.3f}", "±0.03")
        with col_err3:
            st.metric("MAE - Federated", "0.27", "±0.02")
        st.caption("🔧 *Federated metrics đang là placeholder – sẽ cập nhật từ FL logs.*")
    else:
        model_acc = {
            "Naive (trung bình)": 0.65,
            "Local (hồi quy)": 0.78,
            "Federated (FL)": 0.84
        }
        df_models = pd.DataFrame.from_dict(model_acc, orient='index', columns=['Độ chính xác'])
        st.bar_chart(df_models)
        col_err1, col_err2, col_err3 = st.columns(3)
        with col_err1:
            st.metric("MAE - Naive", "0.42", "±0.05")
        with col_err2:
            st.metric("MAE - Local", "0.31", "±0.03")
        with col_err3:
            st.metric("MAE - Federated", "0.27", "±0.02")
        st.caption("🔧 *Đang hiển thị dữ liệu mẫu. Chạy local_models.py để có kết quả thực.*")

st.divider()

# ===== HÀNG 2.5: FL TRAINING MONITOR =====
st.header("📉 FL Training Monitor")
if fl_logs is not None and not fl_logs.empty:
    st.subheader("Loss qua các vòng FL")
    st.line_chart(fl_logs.set_index('round')[['loss']])
    if 'mae' in fl_logs.columns and 'rmse' in fl_logs.columns:
        st.subheader("MAE và RMSE qua các vòng FL")
        st.line_chart(fl_logs.set_index('round')[['mae', 'rmse']])
    last_round = fl_logs['round'].iloc[-1]
    last_loss = fl_logs['loss'].iloc[-1]
    st.metric("Số vòng đã chạy", last_round)
    st.metric("Loss cuối cùng", f"{last_loss:.4f}")
    st.caption("✅ *Dữ liệu được lấy từ results/fl_round_logs.csv*")
else:
    st.info("📭 Chưa có dữ liệu FL logs. Hãy chạy Flower server và đảm bảo file results/fl_round_logs.csv được tạo.")
    st.caption("🔧 *Sẽ hiển thị biểu đồ loss, MAE, RMSE khi có log*")

st.divider()

# ===== HÀNG 3: TỐI ƯU HÓA & ĐIỀU KHIỂN =====
st.header("⚡ Tối ưu hóa & điều khiển")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Chỉ số năng lượng tổng quan")
    if houses_data:
        df_house = houses_data['house1'].iloc[-24:]
        total_today = df_house['main'].sum()
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Tổng tiêu thụ hôm nay", f"{total_today:.2f} (scaled)", "-8%")
        with col_metric2:
            st.metric("Tiết kiệm dự kiến", "...", "+5%")
        with col_metric3:
            st.metric("Thiết bị đang hoạt động", "...", "-2")
        st.caption("🔧 *Tiết kiệm và số thiết bị sẽ cập nhật từ module optimization của Pô*")
    else:
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Tổng tiêu thụ hôm nay", "124 kWh", "-8%")
        with col_metric2:
            st.metric("Tiết kiệm dự kiến", "18 kWh", "+5%")
        with col_metric3:
            st.metric("Thiết bị đang hoạt động", "9", "-2")
    
    st.caption("So sánh tiêu thụ thực tế (xanh) và dự đoán (cam) - 24h qua")
    pred_house = load_predictions("house1")
    if pred_house is not None:
        df_plot = pred_house.tail(24).copy()
        df_plot.set_index('timestamp', inplace=True)
        df_compare = df_plot[['actual', 'predicted']].rename(columns={'actual': 'Thực tế', 'predicted': 'Dự đoán'})
        st.line_chart(df_compare)
        st.caption("🔧 *Dự đoán từ linear regression (có thể thay bằng kết quả từ FL)*")
    elif 'house1' in houses_data:
        df_house = houses_data['house1'].iloc[-24:]
        actual = df_house['main'].values
        predicted = df_house['lag_1'].fillna(method='bfill').values
        df_compare = pd.DataFrame({
            'Giờ': range(24),
            'Thực tế': actual,
            'Dự đoán': predicted
        })
        st.line_chart(df_compare.set_index('Giờ'))
        st.caption("🔧 *Dự đoán hiện là lag_1 (placeholder) – sẽ thay bằng kết quả từ mô hình của Thành*")
    else:
        actual = np.random.normal(5, 1, 24)
        predicted = np.random.normal(4.8, 1, 24)
        df_compare = pd.DataFrame({
            'Giờ': range(24),
            'Thực tế': actual,
            'Dự đoán': predicted
        })
        st.line_chart(df_compare.set_index('Giờ'))

with row3_col2:
    st.subheader("Mức độ ưu tiên thiết bị khi quá tải")
    priority_data = {
        "Thiết bị": ["Điều hòa", "Tủ lạnh", "Máy giặt", "Bình nóng lạnh", "Đèn"],
        "Mức ưu tiên (0-1)": [0.9, 0.8, 0.5, 0.4, 0.2]
    }
    df_priority = pd.DataFrame(priority_data)
    for _, row in df_priority.iterrows():
        st.text(f"{row['Thiết bị']} ({int(row['Mức ưu tiên (0-1)']*100)}%)")
        st.progress(row['Mức ưu tiên (0-1)'])
    st.caption("🔧 *Các mức ưu tiên và cảnh báo dưới đây là placeholder – sẽ tích hợp từ Pô*")
    
    if np.random.random() > 0.7:
        st.error("⚠️ Cảnh báo: Dự báo quá tải trong 2 giờ tới! Đã kích hoạt giảm tải.")
    else:
        st.success("✅ Hệ thống đang hoạt động bình thường, không có nguy cơ quá tải.")

st.divider()

# ===== HÀNG 4: ĐIỀU KHIỂN & BẢO MẬT =====
st.header("🛠️ Điều khiển & Bảo mật")
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.subheader("Điều khiển thiết bị (mô phỏng)")
    with st.expander("Điều khiển nhanh", expanded=True):
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Bật Điều hòa", use_container_width=True):
                st.success("Điều hòa đã BẬT")
        with col_btn2:
            if st.button("Tắt Điều hòa", use_container_width=True):
                st.warning("Điều hòa đã TẮT")
        with col_btn3:
            if st.button("Bật chế độ tiết kiệm", use_container_width=True):
                st.info("Chế độ tiết kiệm đã được kích hoạt")
        st.slider("Ngưỡng cảnh báo quá tải (kW)", 0.0, 20.0, 10.0, 0.5)
        st.caption("Trạng thái thiết bị hiện tại:")
        status_data = {
            "Thiết bị": ["Điều hòa", "Tủ lạnh", "Máy giặt", "Bình nóng lạnh", "Đèn phòng khách"],
            "Trạng thái": ["Bật", "Bật", "Tắt", "Tắt", "Bật"],
            "Công suất (W)": [1200, 150, 500, 2000, 60]
        }
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)
    st.caption("🔧 *Phần điều khiển hiện là mô phỏng – sẽ tích hợp với module optimization của Pô*")

with row4_col2:
    st.subheader("Tổng quan về quyền riêng tư")
    st.info(
        "🔒 **Bảo vệ dữ liệu bằng Federated Learning**\n\n"
        "- Mỗi hộ gia đình huấn luyện mô hình trên dữ liệu cục bộ của chính mình.\n"
        "- **Không có dữ liệu thô nào** rời khỏi nhà thông minh.\n"
        "- Chỉ các trọng số mô hình được gửi về máy chủ trung tâm để tổng hợp.\n"
        "- Quá trình này đảm bảo dữ liệu cá nhân không bị lộ ra ngoài."
    )
    st.caption("📌 *Nội dung privacy đã hoàn chỉnh, sẵn sàng cho báo cáo và demo.*")

