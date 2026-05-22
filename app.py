import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, date

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="E-Commerce Customer Analytics Dashboard", page_icon="🛍️", layout="wide")

st.title("🛍️ Dual-Stage Customer Segmentation & Repurchase Prediction Framework")
st.markdown("""
Hệ thống tự động tiếp nhận **Dữ liệu giao dịch thô (Raw Data)**, tính toán bộ chỉ số **RFMT**, 
thực hiện phân cụm khách hàng và dự báo xác suất quay lại tái mua hàng (Repurchase Prediction).
""")
st.write("---")

# ==========================================
# 2. TẢI VÀ BÓC TÁCH FILE .PKL TỰ ĐỘNG
# ==========================================
MODEL_PATH = "src/hybrid_customer_segmentation_prediction_model.pkl"

@st.cache_resource
def load_all_components():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

pkl_obj = load_all_components()
scaler, kmeans, prediction_model = None, None, None

st.sidebar.header("⚙️ Trạng thái Hệ thống")
if pkl_obj is None:
    st.sidebar.error(f"❌ Không tìm thấy file model tại '{MODEL_PATH}'")
elif isinstance(pkl_obj, dict):
    # Lọc tự động các mô hình từ Dictionary
    for k, v in pkl_obj.items():
        k_lower = k.lower()
        if 'scale' in k_lower: 
            scaler = v
        elif 'mean' in k_lower or 'cluster' in k_lower: 
            kmeans = v
        elif 'model' in k_lower or 'xgb' in k_lower or 'class' in k_lower or 'pred' in k_lower: 
            prediction_model = v
            
    if None not in [scaler, kmeans, prediction_model]:
        st.sidebar.success("✅ Đã kết nối đủ 3 mô hình: Scaler, K-Means, Classifier!")
    else:
        st.sidebar.warning(f"⚠️ Nhận diện thiếu thành phần. Các keys hiện có: {list(pkl_obj.keys())}")

# --- CHIA TAB GIAO DIỆN ---
tab1, tab2 = st.tabs(["👤 Phân tích 1 Khách hàng (Nhập Raw Data)", "📊 Chạy Hàng Loạt (Tải File CSV)"])

# ==========================================
# TAB 1: NHẬP RAW DATA CHO 1 KHÁCH HÀNG
# ==========================================
with tab1:
    st.header("Sổ cái giao dịch (Transaction Ledger) - Đơn lẻ")
    st.write("Nhập lịch sử các lần mua hàng của một khách hàng. Hệ thống sẽ tự động tổng hợp ra RFMT, gán Cụm và Dự báo.")
    
    col_date, col_empty = st.columns([1, 3])
    with col_date:
        analysis_date = st.date_input("📅 Chọn Ngày chốt sổ (Analysis Date):", date.today())
    
    st.write("**Bảng lịch sử mua hàng:** *(Bấm 'Add row' ở góc dưới bảng để thêm các lần mua khác nhau)*")
    
    # Bảng nhập liệu động cho Raw Data
    df_init = pd.DataFrame({
        "Ngày_Mua_Hàng": [date.today().strftime("%Y-%m-%d")],
        "Số_Tiền_Chi_Tiêu": [250.0]
    })
    edited_df = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)
    
    if st.button("🔍 Phân tích Khách hàng này", type="primary"):
        if None in [scaler, kmeans, prediction_model]:
            st.error("⚠️ Hệ thống chưa tải đủ mô hình. Vui lòng kiểm tra lại.")
        else:
            try:
                # 1. TÍNH TOÁN R, F, M, T
                edited_df['Ngày_Mua_Hàng'] = pd.to_datetime(edited_df['Ngày_Mua_Hàng'])
                max_date = edited_df['Ngày_Mua_Hàng'].max()
                min_date = edited_df['Ngày_Mua_Hàng'].min()
                
                f_val = len(edited_df)
                m_val = edited_df['Số_Tiền_Chi_Tiêu'].sum()
                r_val = max(0, (pd.to_datetime(analysis_date) - max_date).days)
                t_val = (max_date - min_date).days
                
                st.write("---")
                st.subheader("1️⃣ Khách hàng này có chỉ số RFMT là:")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Recency", f"{r_val} ngày")
                metric_cols[1].metric("Frequency", f"{f_val} đơn")
                metric_cols[2].metric("Monetary", f"£{m_val:.2f}")
                metric_cols[3].metric("Time T", f"{t_val} ngày")
                
                # 2. TRUYỀN VÀO MÔ HÌNH (LƯU Ý: TÊN CỘT PHẢI KHỚP VỚI PKL FILE)
                input_data = pd.DataFrame([[r_val, f_val, m_val, t_val]], columns=['Recency', 'Frequency', 'Monetary', 'T'])
                
                # Chuẩn hóa -> Phân cụm -> Dự báo
                scaled_data = scaler.transform(input_data)
                cluster_id = kmeans.predict(scaled_data)[0]
                pred = prediction_model.predict(scaled_data)[0]
                
                st.subheader("2️⃣ Kết quả Dự báo AI (Cluster & Repurchase)")
                res_c1, res_c2 = st.columns(2)
                res_c1.info(f"🏷️ **Phân cụm Khách hàng:** Thuộc Nhóm (Cluster ID): **{cluster_id}**")
                
                if pred == 1:
                    res_c2.success("🔮 **Dự báo:** KHÁCH HÀNG SẼ QUAY LẠI TRONG 30 NGÀY TỚI (Will Return) 🎉")
                else:
                    res_c2.error("🚨 **Dự báo:** KHÁCH HÀNG SẼ RỜI BỎ (Not Return) 📉")
                    
            except Exception as e:
                st.error(f"❌ Có lỗi tính toán: {e}. Hãy đảm bảo nhập đúng ngày (YYYY-MM-DD) và số tiền.")


# ==========================================
# TAB 2: XỬ LÝ HÀNG LOẠT (FILE RAW CSV)
# ==========================================
with tab2:
    st.header("Xử lý Luồng dữ liệu giao dịch thô (Raw Transaction Data)")
    st.markdown("""
    **Cấu trúc file CSV tải lên bắt buộc phải có 3 cột:**
    `CustomerID` | `InvoiceDate` | `Total_Amount`
    """)

    uploaded_file = st.file_uploader("Tải lên file Giao dịch thô (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("📊 Bản xem trước dữ liệu thô vừa tải lên:")
        st.dataframe(raw_df.head(5), use_container_width=True)
        
        required = {'CustomerID', 'InvoiceDate', 'Total_Amount'}
        if not required.issubset(raw_df.columns):
            st.error("❌ File thiếu các cột bắt buộc: CustomerID, InvoiceDate, Total_Amount")
        elif None in [scaler, kmeans, prediction_model]:
            st.error("❌ Hệ thống chưa tải đủ mô hình.")
        else:
            with st.spinner("⏳ Đang tự động chuyển đổi Raw Data thành RFMT và chạy dự báo..."):
                try:
                    # 1. BIẾN ĐỔI RAW DATA THÀNH RFMT
                    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])
                    max_date = raw_df['InvoiceDate'].max()
                    
                    rfmt_df = raw_df.groupby('CustomerID').agg({
                        'InvoiceDate': lambda x: (max_date - x.max()).days,
                        'CustomerID': 'count',
                        'Total_Amount': 'sum'
                    }).rename(columns={'InvoiceDate': 'Recency', 'CustomerID': 'Frequency', 'Total_Amount': 'Monetary'})
                    
                    rfmt_df['T'] = raw_df.groupby('CustomerID')['InvoiceDate'].agg(lambda x: (x.max() - x.min()).days)
                    
                    # Bảng kết quả tổng hợp
                    final_output = rfmt_df.copy().reset_index()
                    
                    # 2. CHẠY MÔ HÌNH DỰ BÁO
                    # Đảm bảo thứ tự cột đúng như file .pkl yêu cầu: ['Recency', 'Frequency', 'Monetary', 'T']
                    features = rfmt_df[['Recency', 'Frequency', 'Monetary', 'T']]
                    features_scaled = scaler.transform(features)
                    
                    # Gán Cluster và Dự báo
                    final_output['Cluster_ID'] = kmeans.predict(features_scaled)
                    predictions = prediction_model.predict(features_scaled)
                    final_output['Dự_Báo_Tương_Lai'] = np.where(predictions == 1, 'Sẽ tái mua (Will Return)', 'Nguy cơ rời bỏ (Churn)')
                    
                    st.success("🎉 Đã hoàn tất xử lý! Dưới đây là Bảng kết quả tổng hợp bao gồm RFMT, Cluster ID và Dự báo:")
                    st.dataframe(final_output, use_container_width=True)
                    
                    # Nút tải xuống
                    st.download_button(
                        label="📥 Tải xuống Bảng kết quả hoàn chỉnh (CSV)",
                        data=final_output.to_csv(index=False).encode('utf-8'),
                        file_name="hybrid_framework_predictions_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý dữ liệu: {e}")
