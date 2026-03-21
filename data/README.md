⚡ REDD Dataset - Data Preprocessing Pipeline
Đây là data pipeline cho bộ dữ liệu tiêu thụ điện năng REDD (Reference Energy Disaggregation Data Set).
File data_pipeline.py sẽ đọc raw data từ nhiều ngôi nhà (houses), làm sạch, đồng bộ hóa các đặc trưng (features), chuẩn hóa và biến đổi thành dạng Time-series.


🛠 Yêu cầu hệ thống (Dependencies)
Cài đặt thư viện sau trước khi chạy:
# pip install pandas numpy scikit-learn


📂 Cấu trúc thư mục
📦 REDD_DATASET/
 ┣ 📂 redd/                      # Chứa dữ liệu thô (vd: redd_house1_1.csv, redd_house1_2.csv...)
 ┣ 📜 data_pipeline.py           # File mã nguồn tiền xử lý dữ liệu
 ┗ 📜 README.md                  # File hướng dẫn 

Lưu ý: Folder cleaned_data/ sẽ được tự động tạo ra sau khi chạy. 


📊 Kết quả đầu ra (Output)
Dùng lệnh sau để chạy:
# python data_pipeline.py

Hệ thống sẽ: 
1. Tạo folder cleaned_data/ 
2. Xuất ra 5 file CSV chứa dữ liệu đã làm sạch cho từng nhà (VD: house1_hourly_clean.csv)
3. Tạo ra ma trận dữ liệu với kích thước chuẩn hóa đồng đều giữa các house:
    + Kích thước X(số lượng mẫu, 24, 9) -> Tương ứng với (Samples, Timesteps, Features).
    + Kích thước y: (số lượng mẫu,)