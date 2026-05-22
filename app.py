import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="CRM Analytics & Prediction", page_icon="🛍️", layout="wide")

st.title("🛍️ Hệ thống Tra cứu CRM & Dự báo Hành vi Khách hàng")
st.markdown("Hệ thống tự động tra cứu lịch sử giao dịch thô, tổng hợp RFMT và chạy dự báo bằng mô hình Hybrid.")
st.write("---")

# ==========================================
# 2. TẢI VÀ BÓC TÁCH FILE .PKL MỚI
# ==========================================
# Cập nhật đúng tên file mới bạn vừa tải lên
MODEL_PATH = "hybrid_customer_segmentation_prediction_model (1).pkl" 

@st.cache_resource
def load_models():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return str(e)

pkl_obj = load_models()
scaler, kmeans, prediction_model = None, None, None

if isinstance(pkl_obj, dict):
    for k, v in pkl_obj.items():
        k_lower = k.lower()
        if 'scale' in k_lower: scaler = v
        elif 'mean' in k_lower or 'cluster' in k_lower: kmeans = v
        elif 'model' in k_lower or 'xgb' in k_lower or 'class' in k_lower or 'pred' in k_lower: prediction_model = v

# ==========================================
# 3. HÀM TỰ ĐỘNG THÍCH ỨNG BIẾN (CHỐNG SẬP WEB)
# ==========================================
def predict_xgboost_smart(xgb_model, rfmt_df, scaled_rfmt, cluster_ids):
    """
    Hàm này tự động dò xem XGBoost cần bao nhiêu biến và tự điều chỉnh đầu vào:
    - Nếu cần 4 biến: Chỉ đưa Scaled RFMT.
    - Nếu cần 5 biến: Đưa Scaled RFMT + Cột Cluster ID.
    - Nếu cần 10 biến: Tự động One-hot Encoding (RFMT + Cluster_0 -> Cluster_5).
    """
    try:
        expected_features = xgb_model.n_features_in_
    except AttributeError:
        try:
            expected_features = len(xgb_model.get_booster().feature_names)
        except:
            expected_features = 4 # Mặc định an toàn
            
    if expected_features == 4:
        return xgb_model.predict(scaled_rfmt)
        
    elif expected_features == 5:
        # Ghép thêm cột Cluster_ID vào làm biến thứ 5
        input_5 = np.column_stack((scaled_rfmt, cluster_ids))
        return xgb_model.predict(input_5)
        
    else:
        # Trường hợp >5 biến (Sử dụng One-Hot Encoding như lỗi 10 vs 4)
        try:
            expected_cols = xgb_model.feature_names_in_
        except:
            expected_cols = xgb_model.get_booster().feature_names
            
        df_xgb = pd.DataFrame(0, index=np.arange(len(rfmt_df)), columns=expected_cols)
        
        # Bơm 4 biến RFMT gốc
        for col in ['Recency', 'Frequency', 'Monetary', 'T']:
            if col in df_xgb.columns:
                df_xgb[col] = rfmt_df[col].values
                
        # Bơm One-Hot Encoding cho Cluster
        for i, cid in enumerate(cluster_ids):
            cluster_col = f'Cluster_{cid}'
            if cluster_col in df_xgb.columns:
                df_xgb.loc[i, cluster_col] = 1
                
        return xgb_model.predict(df_xgb)

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.sidebar.header("📂 Nạp Cơ sở dữ liệu")
st.sidebar.write("Tải file dữ liệu giao dịch thô (Database) lên đây để hệ thống tra cứu.")
db_file = st.sidebar.file_uploader("Tải DB (.csv)", type=["csv"])

if db_file is not None:
    db_df = pd.read_csv(db_file)
    db_df['InvoiceDate'] = pd.to_datetime(db_df['InvoiceDate'])
    max_date_db = db_df['InvoiceDate'].max()
    
    tab1, tab2 = st.tabs(["🔍 Tra cứu Khách hàng (CRM)", "📊 Quét hàng loạt toàn bộ DB"])
    
    # --- TAB 1: TRA CỨU TỪNG NGƯỜI ---
    with tab1:
        st.subheader("Nhập ID để phân tích hành vi khách hàng")
        search_id = st.text_input("👤 Mã Khách Hàng (CustomerID):", placeholder="VD: 12345")
        
        if search_id:
            try:
                search_id_clean = float(search_id) if db_df['CustomerID'].dtype in ['int64', 'float64'] else str(search_id)
                user_history = db_df[db_df['CustomerID'] == search_id_clean]
                
                if user_history.empty:
                    st.warning(f"⚠️ Không tìm thấy giao dịch nào của mã '{search_id}'.")
                else:
                    st.success(f"✅ Đã tìm thấy khách hàng {search_id}!")
                    
                    st.write("**🛒 Lịch sử Giao dịch:**")
                    st.dataframe(user_history.sort_values(by='InvoiceDate', ascending=False), use_container_width=True)
                    
                    # Tính toán RFMT
                    user_max_date = user_history['InvoiceDate'].max()
                    user_min_date = user_history['InvoiceDate'].min()
                    
                    r_val = (max_date_db - user_max_date).days
                    f_val = len(user_history)
                    m_val = user_history['Total_Amount'].sum()
                    t_val = (user_max_date - user_min_date).days
                    
                    cols = st.columns(4)
                    cols[0].metric("Recency (R)", f"{r_val} ngày")
                    cols[1].metric("Frequency (F)", f"{f_val} đơn")
                    cols[2].metric("Monetary (M)", f"£{m_val:,.2f}")
                    cols[3].metric("Time T (T)", f"{t_val} ngày")
                    
                    if None not in [scaler, kmeans, prediction_model]:
                        # Tạo DataFrame 1 dòng
                        rfmt_df = pd.DataFrame([[r_val, f_val, m_val, t_val]], columns=['Recency', 'Frequency', 'Monetary', 'T'])
                        scaled_data = scaler.transform(rfmt_df)
                        cluster_id = kmeans.predict(scaled_data)
                        
                        # Gọi hàm dự báo thông minh
                        pred = predict_xgboost_smart(prediction_model, rfmt_df, scaled_data, cluster_id)[0]
                        
                        st.write("---")
                        c1, c2 = st.columns(2)
                        c1.info(f"🏷️ **Phân khúc:** Thuộc **Nhóm (Cluster) {cluster_id[0]}**")
                        
                        if pred == 1:
                            c2.success("🔮 **Dự báo:** SẼ QUAY LẠI TÁI MUA (Will Return) 🎉")
                        else:
                            c2.error("🚨 **Dự báo:** KHÔNG QUAY LẠI (Nguy cơ rời bỏ) 📉")
                    else:
                        st.error("Lỗi: File model chưa đầy đủ thành phần.")
                        
            except ValueError:
                st.error("Vui lòng nhập CustomerID hợp lệ.")
                
    # --- TAB 2: QUÉT HÀNG LOẠT ---
    with tab2:
        st.subheader("Chạy dự báo cho toàn bộ Cơ sở dữ liệu")
        if st.button("🚀 Kích hoạt Quét hệ thống", type="primary"):
            with st.spinner("Đang tính toán RFMT và chạy mô hình cho toàn bộ khách hàng..."):
                try:
                    # Tính RFMT cho tất cả
                    rfmt_all = db_df.groupby('CustomerID').agg({
                        'InvoiceDate': lambda x: (max_date_db - x.max()).days,
                        'CustomerID': 'count',
                        'Total_Amount': 'sum'
                    }).rename(columns={'InvoiceDate': 'Recency', 'CustomerID': 'Frequency', 'Total_Amount': 'Monetary'})
                    rfmt_all['T'] = db_df.groupby('CustomerID')['InvoiceDate'].agg(lambda x: (x.max() - x.min()).days)
                    
                    final_df = rfmt_all.copy().reset_index()
                    features = rfmt_all[['Recency', 'Frequency', 'Monetary', 'T']]
                    
                    # Mô hình AI
                    features_scaled = scaler.transform(features)
                    final_df['Cluster_ID'] = kmeans.predict(features_scaled)
                    
                    # Dự báo thông minh
                    predictions = predict_xgboost_smart(prediction_model, rfmt_all, features_scaled, final_df['Cluster_ID'].values)
                    final_df['Dự_Báo'] = np.where(predictions == 1, 'Sẽ tái mua', 'Rời bỏ')
                    
                    st.success("🎉 Đã hoàn tất! Dưới đây là kết quả:")
                    st.dataframe(final_df, use_container_width=True)
                    
                    st.download_button("📥 Tải File Báo Cáo (CSV)", final_df.to_csv(index=False).encode('utf-8'), "crm_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
else:
    st.info("👈 **Vui lòng tải file lịch sử giao dịch (Raw Data.csv) ở thanh bên trái để bắt đầu.**")
