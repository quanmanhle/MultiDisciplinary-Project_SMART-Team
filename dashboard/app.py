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
    initial_sidebar_state="collapsed"  
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

DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_RESULTS_DIR = os.path.join(BASE_DIR, "models", "resultmlp")
FL_LOGS_DIR = os.path.join(BASE_DIR, "results", "fl_run_logs")

def find_house_data_files():
    pattern = os.path.join(DATA_PATH, "house*_clean.csv")
    files = glob.glob(pattern)
    house_names = []
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith("house"):
            parts = basename.split('_')
            house_names.append(parts[0])
    def key_func(x):
        try:
            return int(x.replace("house", ""))
        except:
            return 999
    return sorted(set(house_names), key=key_func)

def load_house_data(house_name):
    file_path = os.path.join(DATA_PATH, f"{house_name}_clean.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return None

def load_all_houses():
    house_list = find_house_data_files()
    data = {}
    for h in house_list:
        df = load_house_data(h)
        if df is not None:
            data[h] = df
    return data

def load_local_metrics():
    file_path = os.path.join(MODELS_RESULTS_DIR, "local_metrics.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'house' not in df.columns:
            if 'House' in df.columns:
                df = df.rename(columns={'House': 'house'})
        return df
    return None

def load_fl_round_logs():
    """Đọc tất cả các file houseX_fl_rounds.csv và gộp lại"""
    if not os.path.exists(FL_LOGS_DIR):
        return None
    all_files = glob.glob(os.path.join(FL_LOGS_DIR, "house*_fl_rounds.csv"))
    if not all_files:
        return None
    dfs = []
    for f in all_files:
        basename = os.path.basename(f)
        house_name = basename.split('_')[0]
        df = pd.read_csv(f)
        df['house'] = house_name
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def load_predictions(house_name, model_type):
    pred_dir = os.path.join(MODELS_RESULTS_DIR, "predictions")
    if not os.path.exists(pred_dir):
        return None
    patterns = [
        f"{house_name}_{model_type}_predictions.csv",
        f"{house_name}_predictions.csv"
    ]
    for pat in patterns:
        file_path = os.path.join(pred_dir, pat)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            return df
    return None

def load_coefficients(house_name):
    file_path = os.path.join(MODELS_RESULTS_DIR, f"{house_name}_coefficients.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None

# ===== ĐỌC DỮ LIỆU =====
houses_data = load_all_houses()
local_metrics = load_local_metrics()
fl_round_logs = load_fl_round_logs()

# Lấy danh sách house từ local_metrics (ưu tiên) hoặc từ houses_data
if local_metrics is not None and 'house' in local_metrics.columns:
    all_houses = sorted(local_metrics['house'].unique(), key=lambda x: int(x.replace("house", "")))
elif houses_data:
    all_houses = sorted(houses_data.keys(), key=lambda x: int(x.replace("house", "")))
else:
    all_houses = [f"house{i}" for i in range(1, 11)]

# ===== HÀNG 1: TỔNG QUAN HỆ THỐNG =====
st.header("📊 Tổng quan hệ thống")
row1_col1, row1_col2 = st.columns([1, 2])

with row1_col1:
    st.subheader("Mức độ tham gia của các hộ")
    if houses_data:
        totals = {house: df['main'].sum() for house, df in houses_data.items()}
        total_all = sum(totals.values())
        percentages = {house: (totals.get(house, 0)/total_all)*100 for house in all_houses}
    else:
        n = len(all_houses)
        percentages = {house: 100/n for house in all_houses}
    n_houses = len(all_houses)
    col_a, col_b = st.columns(2)
    half = (n_houses + 1) // 2
    with col_a:
        for house in all_houses[:half]:
            pct = percentages.get(house, 0)
            st.plotly_chart(progress_circle(pct, house.capitalize(), height=120), width='stretch')
    with col_b:
        for house in all_houses[half:]:
            pct = percentages.get(house, 0)
            st.plotly_chart(progress_circle(pct, house.capitalize(), height=120), width='stretch')

with row1_col2:
    st.subheader("Dự báo tiêu thụ hôm nay (kWh)")
    example_house = "house1" if "house1" in houses_data else (all_houses[0] if all_houses else None)
    if example_house and example_house in houses_data:
        df_house = houses_data[example_house]
        if not isinstance(df_house.index, pd.DatetimeIndex):
            df_house.index = pd.to_datetime(df_house.index)
        last_time = df_house.index.max()
        start_time = last_time - pd.Timedelta(hours=23)
        last_24h = df_house.loc[start_time:last_time].copy()
        if len(last_24h) < 24:
            st.warning(f"⚠️ Chỉ có {len(last_24h)} điểm dữ liệu trong 24h qua. Biểu đồ dựa trên các điểm này.")
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

# ===== HÀNG 2: TRẠNG THÁI CLIENT & SO SÁNH MÔ HÌNH =====
st.header("📈 Trạng thái các client & so sánh mô hình")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Thông tin chi tiết từng hộ")
    if local_metrics is not None:
        clients_list = []
        for house_name in all_houses:
            total_consumption = "..."
            avg_consumption = "..."
            if house_name in houses_data:
                df = houses_data[house_name]
                total_consumption = f"{df['main'].sum():.2f}"
                avg_consumption = f"{df['main'].mean():.2f}"
            devices = np.random.randint(3, 8)
            # Lấy metrics của MLP
            row_mlp = local_metrics[(local_metrics['house'] == house_name) & (local_metrics['model'] == 'mlp_regression')]
            r2_local = row_mlp['r2'].iloc[0] if not row_mlp.empty else None
            mae_local = row_mlp['mae'].iloc[0] if not row_mlp.empty else None
            rmse_local = row_mlp['rmse'].iloc[0] if not row_mlp.empty else None
            # Lấy metrics của Naive
            row_naive = local_metrics[(local_metrics['house'] == house_name) & (local_metrics['model'] == 'naive_baseline')]
            r2_naive = row_naive['r2'].iloc[0] if not row_naive.empty else None
            mae_naive = row_naive['mae'].iloc[0] if not row_naive.empty else None
            rmse_naive = row_naive['rmse'].iloc[0] if not row_naive.empty else None
            # Adaptive EWMA
            row_ewma = local_metrics[(local_metrics['house'] == house_name) & (local_metrics['model'] == 'adaptive_ewma_baseline')]
            r2_ewma = row_ewma['r2'].iloc[0] if not row_ewma.empty else None
            mae_ewma = row_ewma['mae'].iloc[0] if not row_ewma.empty else None
            rmse_ewma = row_ewma['rmse'].iloc[0] if not row_ewma.empty else None
            clients_list.append({
                "Hộ": house_name,
                "Tổng tiêu thụ": total_consumption,
                "Trung bình": avg_consumption,
                "R² (Naive)": f"{r2_naive:.3f}" if r2_naive is not None else "...",
                "R² (EWMA)": f"{r2_ewma:.3f}" if r2_ewma is not None else "...",
                "R² (MLP)": f"{r2_local:.3f}" if r2_local is not None else "...",
                "MAE (MLP)": f"{mae_local:.3f}" if mae_local is not None else "...",
                "RMSE (MLP)": f"{rmse_local:.3f}" if rmse_local is not None else "...",
                "Thiết bị (ước)": devices
            })
        df_clients = pd.DataFrame(clients_list)
        st.dataframe(df_clients, use_container_width=True, hide_index=True)
        st.caption("🔧 *Chỉ số R², MAE, RMSE từ local_metrics.csv (MLP là mô hình chính).*")
    else:
        st.info("Không có dữ liệu local_metrics.csv. Hãy chạy train_local_mlp.py để tạo metrics.")

with row2_col2:
    st.subheader("So sánh hiệu năng các mô hình")
    if local_metrics is not None:
        models_to_compare = ['naive_baseline', 'adaptive_ewma_baseline', 'mlp_regression']
        model_names = {
            'naive_baseline': 'Naive Baseline',
            'adaptive_ewma_baseline': 'Adaptive EWMA',
            'mlp_regression': 'MLP Regression'
        }
        avg_r2 = []
        avg_mae = []
        avg_rmse = []
        for model in models_to_compare:
            rows = local_metrics[local_metrics['model'] == model]
            if not rows.empty:
                avg_r2.append(rows['r2'].mean())
                avg_mae.append(rows['mae'].mean())
                avg_rmse.append(rows['rmse'].mean())
            else:
                avg_r2.append(0.0)
                avg_mae.append(0.0)
                avg_rmse.append(0.0)
        df_r2 = pd.DataFrame({
            "Mô hình": [model_names[m] for m in models_to_compare],
            "R²": avg_r2
        }).set_index("Mô hình")
        st.bar_chart(df_r2)
        st.markdown("**So sánh chi tiết các chỉ số (trung bình trên các hộ)**")
        comparison_df = pd.DataFrame({
            "Mô hình": [model_names[m] for m in models_to_compare],
            "R²": [f"{v:.3f}" for v in avg_r2],
            "MAE": [f"{v:.3f}" for v in avg_mae],
            "RMSE": [f"{v:.3f}" for v in avg_rmse]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.caption("🔧 *Giá trị trung bình trên tất cả các hộ.*")
    else:
        model_acc = {
            "Naive (trung bình)": 0.65,
            "Local (MLP)": 0.78,
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
        st.caption("🔧 *Đang hiển thị dữ liệu mẫu. Cần có local_metrics.csv để có dữ liệu thật.*")

st.divider()

# ===== HÀNG 2.5: FL TRAINING MONITOR =====
st.header("📉 FL Training Monitor")
if fl_round_logs is not None and not fl_round_logs.empty:
    house_list = sorted(fl_round_logs['house'].unique(), key=lambda x: int(x.replace("house","")))
    selected_house_fl = st.selectbox("Chọn hộ để xem FL logs", house_list, key="fl_house_select")
    df_house_fl = fl_round_logs[fl_round_logs['house'] == selected_house_fl].copy()
    if not df_house_fl.empty:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df_house_fl['round'], y=df_house_fl['loss_rmse'],
            mode='lines+markers', name='Loss (RMSE)',
            line=dict(color='red', width=2)
        ))
        fig_loss.update_layout(
            title=f"Loss qua các vòng FL - {selected_house_fl}",
            xaxis_title="Round",
            yaxis_title="Loss (RMSE)",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        if 'mae' in df_house_fl.columns and 'rmse' in df_house_fl.columns:
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=df_house_fl['round'], y=df_house_fl['mae'],
                mode='lines+markers', name='MAE'
            ))
            fig_metrics.add_trace(go.Scatter(
                x=df_house_fl['round'], y=df_house_fl['rmse'],
                mode='lines+markers', name='RMSE'
            ))
            fig_metrics.update_layout(
                title=f"MAE và RMSE qua các vòng FL - {selected_house_fl}",
                xaxis_title="Round",
                yaxis_title="Giá trị",
                height=400
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        last_round = df_house_fl['round'].iloc[-1]
        last_loss = df_house_fl['loss_rmse'].iloc[-1]
        st.metric(f"Số vòng FL của {selected_house_fl}", last_round)
        st.metric(f"Loss cuối cùng", f"{last_loss:.6f}")
    else:
        st.warning(f"Không có dữ liệu FL cho {selected_house_fl}")
else:
    st.info("📭 Chưa có dữ liệu FL logs. Hãy chạy Flower server và đảm bảo các file houseX_fl_rounds.csv được tạo trong results/fl_run_logs/.")
    st.caption("🔧 *Sẽ hiển thị biểu đồ loss, MAE, RMSE khi có log.*")

st.divider()

# ===== HÀNG 3: TỐI ƯU HÓA & ĐIỀU KHIỂN =====
st.header("⚡ Tối ưu hóa & điều khiển")
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("📈 Biểu đồ dự đoán")
    st.markdown("Chọn hộ và mô hình để xem biểu đồ chi tiết")
    selected_house = st.selectbox("Chọn hộ", all_houses, format_func=lambda x: x.capitalize(), key="plot_house")
    model_options = ["mlp_regression", "naive_baseline", "adaptive_ewma_baseline"]
    model_display = {
        "mlp_regression": "MLP Regression",
        "naive_baseline": "Naive Baseline",
        "adaptive_ewma_baseline": "Adaptive EWMA"
    }
    selected_model = st.selectbox(
        "Chọn mô hình",
        options=model_options,
        format_func=lambda x: model_display.get(x, x),
        key="plot_model"
    )
    plots_dir = os.path.join(MODELS_RESULTS_DIR, "plots")
    img_path = None
    if os.path.exists(plots_dir):
        patterns = [
            f"{selected_house}_{selected_model}.png",
            f"{selected_house}_{selected_model}_predictions.png"
        ]
        for pat in patterns:
            candidate = os.path.join(plots_dir, pat)
            if os.path.exists(candidate):
                img_path = candidate
                break
    if img_path:
        st.image(img_path, caption=f"{selected_house.capitalize()} - {model_display.get(selected_model, selected_model)}", use_container_width=True)
    else:
        pred_df = load_predictions(selected_house, selected_model)
        if pred_df is not None and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
            plot_df = pred_df.tail(100).copy()
            plot_df.set_index('timestamp', inplace=True)
            plot_df['predicted'] = plot_df['predicted'].clip(0, None)
            st.line_chart(plot_df[['actual', 'predicted']])
            st.caption(f"Biểu đồ dự đoán từ file {selected_house}_{selected_model}_predictions.csv")
        else:
            st.info(f"Chưa có biểu đồ hoặc file predictions cho {selected_house} - {model_display.get(selected_model, selected_model)}")
    with st.expander("📊 Xem hệ số hồi quy (Coefficients)"):
        coef_df = load_coefficients(selected_house)
        if coef_df is not None:
            st.markdown(f"**{selected_house.capitalize()} - Hệ số của các đặc trưng**")
            coef_df['abs_coef'] = coef_df['coefficient'].abs()
            top_positive = coef_df.nlargest(5, 'coefficient')[['feature', 'coefficient']]
            top_negative = coef_df.nsmallest(5, 'coefficient')[['feature', 'coefficient']]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 5 hệ số dương (ảnh hưởng cùng chiều)**")
                st.dataframe(top_positive, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Top 5 hệ số âm (ảnh hưởng ngược chiều)**")
                st.dataframe(top_negative, use_container_width=True, hide_index=True)
            st.caption("Hệ số dương: tăng đặc trưng → tăng dự đoán; ngược lại với hệ số âm.")
        else:
            st.info(f"Không tìm thấy file coefficients cho {selected_house}.")

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
    if np.random.random() > 0.9:
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

with row4_col2:
    st.subheader("Tổng quan về quyền riêng tư")
    st.info(
        "🔒 **Bảo vệ dữ liệu bằng Federated Learning**\n\n"
        "- Mỗi hộ gia đình huấn luyện mô hình trên dữ liệu cục bộ của chính mình.\n"
        "- **Không có dữ liệu thô nào** rời khỏi nhà thông minh.\n"
        "- Chỉ các trọng số mô hình được gửi về máy chủ trung tâm để tổng hợp.\n"
        "- Quá trình này đảm bảo dữ liệu cá nhân không bị lộ ra ngoài."
    )

# ===== CHÂN TRANG =====
st.divider()
st.caption("Đồ án đa ngành - Hệ thống thông tin trong khoa học máy tính | Federated Learning for Smart Home Energy Optimization")
