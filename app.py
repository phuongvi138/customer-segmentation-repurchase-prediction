import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="E-Commerce Customer Analytics Dashboard", page_icon="🛍️", layout="wide")

st.title("🛍️ Dual-Stage Customer Segmentation & Repurchase Prediction Framework")
st.markdown("""
Ứng dụng này tự động tiếp nhận **Dữ liệu giao dịch thô (Raw Data)**, tính toán bộ chỉ số **RFMT**, 
thực hiện phân cụm khách hàng và dự báo xác suất quay lại tái mua hàng trong 30 ngày tới.
""")
st.write("---")

# 2. TẢI FILE `.PKL` VÀ TỰ ĐỘNG BÓC TÁCH CÁC THÀNH PHẦN
MODEL_PATH = "src/hybrid_customer_segmentation_prediction_model.pkl"

@st.cache_resource
def load_all_components():
    try:
        pkl_obj = joblib.load(MODEL_PATH)
        return pkl_obj
    except Exception as e:
        st.error(f"⚠️ Không thể tải file model tại '{MODEL_PATH}': {e}")
        return None

pkl_obj = load_all_components()

# Khởi tạo các biến mô hình trống
scaler, kmeans, prediction_model = None, None, None

if pkl_obj is not None and isinstance(pkl_obj, dict):
    st.sidebar.success("✅ Đã kết nối thành công với bộ mô hành Hybrid!")
    # Tự động dò tìm các thành phần dựa trên các key phổ biến bạn cùng nhóm có thể đặt
    scaler = pkl_obj.get('scaler') or pkl_obj.get('standard_scaler')
    kmeans = pkl_obj.get('kmeans') or pkl_obj.get('clustering_model') or pkl_obj.get('cluster')
    prediction_model = pkl_obj.get('model') or pkl_obj.get('xgb') or pkl_obj.get('xgboost') or pkl_obj.get('classifier')

# --- CHIA TAB GIAO DIỆN ---
tab1, tab2 = st.tabs(["🎯 Dự báo nhanh cho 1 Khách hàng", "📊 Tải File Giao Dịch Thô (Raw Data) & Chạy Hàng Loạt"])

# TAB 1: DỰ BÁO NHANH QUA THANH TRƯỢT (Nếu đã biết sẵn RFMT)
with tab1:
    st.header("Dự báo nhanh dựa trên chỉ số RFMT")
    col1, col2, col3, col4 = st.columns(4)
    with col1: r_val = st.slider("Recency (Ngày)", 0, 365, 30)
    with col2: f_val = st.slider("Frequency (Số đơn)", 1, 100, 5)
    with col3: m_val = st.number_input("Monetary (£)", min_value=0.0, value=250.0)
    with col4: t_val = st.slider("Time T (Nhịp điệu - Ngày)", 0, 180, 15)

    if st.button("Chạy dự báo nhanh"):
        if None in [scaler, kmeans, prediction_model]:
            st.error("⚠️ File .pkl thiếu thành phần hoặc chưa bóc tách đúng khóa. Vui lòng kiểm tra cấu trúc file.")
        else:
            input_data = pd.DataFrame([[r_val, f_val, m_val, t_val]], columns=['R', 'F', 'M', 'T'])
            scaled_data = scaler.transform(input_data)
            cluster_id = kmeans.predict(scaled_data)[0]
            
            # Giả sử mô hình phân loại nhận vào dữ liệu đã scale
            pred = prediction_model.predict(scaled_data)[0]
            
            st.write("---")
            c1, c2 = st.columns(2)
            c1.metric("Phân cụm khách hàng (Cluster ID)", f"Cluster {cluster_id}")
            if pred == 1:
                c2.success("🔮 Kết quả: SẼ QUAY LẠI TRONG 30 NGÀY TỚI")
            else:
                c2.error("🚨 Kết quả: KHÔNG QUAY LẠI (Nguy cơ rời bỏ)")

# TAB 2: XỬ LÝ ĐÚNG LUỒNG LOGIC BÀI BÁO (NHẬN FILE RAW DATA)
with tab2:
    st.header("Xử lý Luồng dữ liệu giao dịch thô (Raw Transaction Data)")
    st.markdown("""
    **Yêu cầu cấu trúc file CSV tải lên phải có tối thiểu các cột sau:**
    * `CustomerID`: Mã định danh khách hàng.
    * `InvoiceDate`: Ngày thực hiện giao dịch (Định dạng: YYYY-MM-DD hoặc DD-MM-YYYY).
    * `Total_Amount`: Số tiền của đơn hàng đó (hoặc cột tính tổng bằng `Quantity * UnitPrice`).
    """)

    uploaded_file = st.file_uploader("Tải lên file Giao dịch thô (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("📊 Bản xem trước dữ liệu thô vừa tải lên:")
        st.dataframe(raw_df.head(5), use_container_width=True)
        
        # Kiểm tra cột bắt buộc
        required = {'CustomerID', 'InvoiceDate', 'Total_Amount'}
        if not required.issubset(raw_df.columns):
            st.error("❌ File thiếu các cột bắt buộc: CustomerID, InvoiceDate, Total_Amount")
        elif None in [scaler, kmeans, prediction_model]:
            st.error("❌ Hệ thống chưa bóc tách được đủ 3 thành phần (Scaler, K-Means, XGBoost) từ file .pkl để chạy.")
        else:
            with st.spinner("⏳ Hệ thống đang tự động tính toán chỉ số RFMT và chạy mô hình Hybrid..."):
                try:
                    # --- BƯỚC 1: TỰ ĐỘNG TÍNH TOÁN R, F, M, T (FEATURE ENGINEERING) ---
                    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
                    max_date = raw_df['InvoiceDate'].max()
                    
                    # Tính toán R, F, M
                    rfmt_df = raw_df.groupby('CustomerID').agg({
                        'InvoiceDate': lambda x: (max_date - x.max()).days, # Recency
                        'CustomerID': 'count',                             # Frequency
                        'Total_Amount': 'sum'                              # Monetary
                    }).rename(columns={'InvoiceDate': 'R', 'CustomerID': 'F', 'Total_Amount': 'M'})
                    
                    # Tính toán chỉ số T (Nhịp điệu thời gian tương tác/Tenure)
                    tenure = raw_df.groupby('CustomerID')['InvoiceDate'].agg(lambda x: (x.max() - x.min()).days)
                    rfmt_df['T'] = tenure
                    
                    # Giữ lại bảng RFMT gốc làm kết quả hiển thị (Yêu cầu số 1 của bạn)
                    rfmt_output = rfmt_df.copy().reset_index()
                    
                    # --- BƯỚC 2: CHUẨN HÓA VÀ DỰ BÁO CỤM (Yêu cầu số 2 của bạn) ---
                    features = rfmt_df[['R', 'F', 'M', 'T']]
                    features_scaled = scaler.transform(features)
                    
                    rfmt_output['Cluster_ID'] = kmeans.predict(features_scaled)
                    
                    # --- BƯỚC 3: DỰ BÁO TÁI MUA TRONG 30 NGÀY TỚI (Yêu cầu số 3 của bạn) ---
                    predictions = prediction_model.predict(features_scaled)
                    rfmt_output['Dự_Báo_30_Ngày_Tới'] = np.where(predictions == 1, 'Quay lại (Will Return)', 'Không quay lại (Churn)')
                    
                    # --- HIỂN THỊ KẾT QUẢ CUỐI CÙNG ---
                    st.success("🎉 Đã xử lý và kết xuất dữ liệu thành công theo đúng mô hình của bài báo!")
                    
                    st.subheader("📋 Bảng kết quả định danh và phân tích hành vi khách hàng toàn diện")
                    st.dataframe(rfmt_output, use_container_width=True)
                    
                    # Nút Tải xuống file kết quả tổng hợp
                    st.write("---")
                    csv_bytes = rfmt_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống bảng kết quả tổng hợp (RFMT + Cluster + Dự báo).csv",
                        data=csv_bytes,
                        file_name="hybrid_framework_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình tính toán đặc trưng RFMT từ file thô: {e}")
