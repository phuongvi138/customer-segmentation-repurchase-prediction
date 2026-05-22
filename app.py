import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. GIAO DIỆN CHÍNH & PHONG CÁCH
# ==========================================
st.set_page_config(page_title="CRM Analytics & Prediction Dashboard", page_icon="🛍️", layout="wide")

st.title("🛍️ Phân đoạn Khách hàng & Dự đoán Tái mua hàng")
st.markdown("Ứng dụng tự động chuyển đổi Dữ liệu giao dịch thô (Raw Data) thành chỉ số hành vi **RFMT**, thực hiện phân cụm và dự báo hành vi tương lai bằng mô hình Hybrid.")
st.write("---")

# ==========================================
# 2. TẢI VÀ BÓC TÁCH FILE .PKL CHUẨN XÁC
# ==========================================
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

# Hiển thị trạng thái kết nối mô hình ở thanh bên
st.sidebar.header("⚙️ Cấu hình Hệ thống")
if None not in [scaler, kmeans, prediction_model]:
    st.sidebar.success("✅ Đã kết nối thành công Bộ ba Mô hình AI!")
else:
    st.sidebar.error("❌ Lỗi tải mô hình. Vui lòng kiểm tra lại file .pkl trong thư mục.")

# ==========================================
# 3. THUẬT TOÁN XỬ LÝ THÍCH ỨNG 3 BIẾN CHO XGBOOST
# ==========================================
def predict_xgboost_smart(xgb_model, rfmt_df, scaled_rfmt, cluster_ids):
    """
    Hàm xử lý thông minh thích ứng số lượng biến đầu vào cho XGBoost:
    Nhận diện chính xác mô hình mới yêu cầu 3 features (Recency, Frequency, Monetary).
    """
    try:
        expected_features = xgb_model.n_features_in_
    except AttributeError:
        try:
            expected_features = len(xgb_model.get_booster().feature_names)
        except:
            expected_features = 3 # Định dạng mặc định theo file pkl mới của bạn
            
    if expected_features == 3:
        # Lấy 3 cột đầu tiên của dữ liệu đã chuẩn hóa (tương ứng với Recency, Frequency, Monetary)
        return xgb_model.predict(scaled_rfmt[:, :3])
    elif expected_features == 4:
        return xgb_model.predict(scaled_rfmt)
    elif expected_features == 5:
        input_5 = np.column_stack((scaled_rfmt, cluster_ids))
        return xgb_model.predict(input_5)
    else:
        # Xử lý trường hợp mô hình một-hot nâng cao (10 biến)
        try: expected_cols = xgb_model.feature_names_in_
        except: expected_cols = xgb_model.get_booster().feature_names
        
        df_xgb = pd.DataFrame(0, index=np.arange(len(rfmt_df)), columns=expected_cols)
        for i, col in enumerate(['Recency', 'Frequency', 'Monetary', 'T']):
            if col in df_xgb.columns: df_xgb[col] = scaled_rfmt[:, i]
        for i, cid in enumerate(cluster_ids):
            cluster_col = f'Cluster_{cid}'
            if cluster_col in df_xgb.columns: df_xgb.loc[i, cluster_col] = 1
        return xgb_model.predict(df_xgb)

# ==========================================
# 4. TIẾP NHẬN FILE DỮ LIỆU TỪ SIDEBAR
# ==========================================
st.sidebar.subheader("📂 Nạp Cơ sở dữ liệu")
st.sidebar.write("Tải lên tệp lịch sử hóa đơn bán hàng định dạng `.csv`.")
db_file = st.sidebar.file_uploader("Chọn file dữ liệu giao dịch thô", type=["csv"])

if db_file is not None:
    db_df = pd.read_csv(db_file)
    
    # --- THUẬT TOÁN TỰ ĐỘNG DÒ TÊN CỘT THÔNG MINH (CHỐNG LỖI KEYERROR) ---
    id_col, date_col, amount_col = None, None, None
    for col in db_df.columns:
        col_lower = col.lower().replace("_", "").replace(" ", "")
        if col_lower in ['customerid', 'customer_id', 'id', 'makhachhang', 'makh']: id_col = col
        elif col_lower in ['invoicedate', 'invoice_date', 'date', 'ngay', 'ngayhoadon']: date_col = col
        elif col_lower in ['totalamount', 'amount', 'price', 'revenue', 'sales', 'total', 'sotien']: amount_col = col

    # Trường hợp file có Quantity và UnitPrice thay vì tính sẵn Total_Amount
    if amount_col is None and 'Quantity' in db_df.columns and 'UnitPrice' in db_df.columns:
        db_df['Total_Amount'] = db_df['Quantity'] * db_df['UnitPrice']
        amount_col = 'Total_Amount'

    # Kiểm tra tính hợp lệ của dữ liệu đầu vào
    if None in [id_col, date_col, amount_col]:
        st.error("❌ Định dạng file CSV không tương thích! Đảm bảo file có chứa các thông tin định danh: Mã khách hàng, Ngày giao dịch, và Số tiền hóa đơn.")
    else:
        # Chuẩn hóa định dạng thời gian
        db_df[date_col] = pd.to_datetime(db_df[date_col])
        max_date_db = db_df[date_col].max()
        
        # Chia các phân khu tính năng bằng Tab trực quan
        tab1, tab2 = st.tabs(["🔍 Tra cứu Khách hàng (CRM)", "📊 Quét hệ thống hàng loạt"])
        
        # ------------------------------------------
        # TAB 1: GIAO DIỆN TRA CỨU CHI TIẾT TỪNG ID
        # ------------------------------------------
        with tab1:
            st.subheader("Hồ sơ Hành vi & Dự báo Đơn lẻ")
            search_id = st.text_input("👤 Nhập Mã số Khách hàng (CustomerID) cần kiểm tra:", placeholder="Ví dụ: 17850, 13047...")
            
            if search_id:
                # Xử lý đồng bộ kiểu dữ liệu cho mã khách hàng
                if db_df[id_col].dtype in ['int64', 'float64']:
                    try: search_id_clean = float(search_id)
                    except ValueError: search_id_clean = search_id
                else:
                    search_id_clean = str(search_id)
                
                user_history = db_df[db_df[id_col] == search_id_clean]
                
                if user_history.empty:
                    st.warning(f"⚠️ Không tìm thấy bất kỳ giao dịch nào tương ứng với mã khách hàng '{search_id}' trong hệ thống.")
                else:
                    st.success(f"🎉 Kết nối hồ sơ khách hàng {search_id} thành công!")
                    
                    # Tính toán dữ liệu RFMT thực tế của khách hàng
                    user_max_date = user_history[date_col].max()
                    user_min_date = user_history[date_col].min()
                    
                    r_val = (max_date_db - user_max_date).days
                    f_val = len(user_history)
                    m_val = user_history[amount_col].sum()
                    t_val = (user_max_date - user_min_date).days
                    
                    # Hiển thị các chỉ số cốt lõi dưới dạng các thẻ Metric lớn giống như dashboard chuyên nghiệp
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Recency (R - Gần nhất)", f"{r_val} ngày")
                    m_col2.metric("Frequency (F - Tần suất)", f"{f_val} lần")
                    m_col3.metric("Monetary (M - Giá trị)", f"£{m_val:,.2f}")
                    m_col4.metric("Time T (T - Vòng đời)", f"{t_val} ngày")
                    
                    # Dự báo thông qua bộ ba AI
                    if None not in [scaler, kmeans, prediction_model]:
                        rfmt_single = pd.DataFrame([[r_val, f_val, m_val, t_val]], columns=['Recency', 'Frequency', 'Monetary', 'T'])
                        scaled_single = scaler.transform(rfmt_single)
                        cluster_single = kmeans.predict(scaled_single)
                        
                        pred_single = predict_xgboost_smart(prediction_model, rfmt_single, scaled_single, cluster_single)[0]
                        
                        st.write("---")
                        st.subheader("🔮 Kết quả Đánh giá Phân tích từ AI")
                        res_col1, res_col2 = st.columns(2)
                        
                        res_col1.info(f"🏷️ **Phân khúc Thị trường:** Khách hàng thuộc nhóm **[Cluster {cluster_single[0]}]**")
                        if pred_single == 1:
                            res_col2.success("🔮 **Dự báo mô hình (30 ngày tới):** KHÁCH HÀNG SẼ QUAY LẠI MUA HÀNG (Will Return) 🎉")
                        else:
                            res_col2.error("🚨 **Dự báo mô hình (30 ngày tới):** NGUY CƠ RỜI BỎ HỆ THỐNG (Churn Risk) 📉")
                    
                    # Bảng lịch sử mua hàng chi tiết nằm ở dưới cùng
                    st.write("---")
                    st.subheader("🛒 Chi tiết Nhật ký Giao dịch")
                    st.dataframe(user_history.sort_values(by=date_col, ascending=False), use_container_width=True)
                    
        # ------------------------------------------
        # TAB 2: GIAO DIỆN XỬ LÝ QUÉT TOÀN BỘ FILE DB
        # ------------------------------------------
        with tab2:
            st.subheader("Phân tích tự động danh sách toàn bộ Cơ sở dữ liệu")
            st.write("Hệ thống sẽ quét qua toàn bộ danh sách hóa đơn, tự gom nhóm từng khách hàng, tính toán chỉ số hành vi và dự báo tự động.")
            
            if st.button("🚀 Kích hoạt tiến trình quét hàng loạt", type="primary"):
                with st.spinner("Hệ thống AI đang xử lý tính toán cấu trúc dữ liệu hành vi toàn cục..."):
                    try:
                        # 1. Gom nhóm tự động tính toán nhanh RFMT loại bỏ multi-index xung đột gán cột
                        rfmt_all = db_df.groupby(id_col).agg(
                            Recency=(date_col, lambda x: (max_date_db - x.max()).days),
                            Frequency=(date_col, 'count'),
                            Monetary=(amount_col, 'sum'),
                            T=(date_col, lambda x: (x.max() - x.min()).days)
                        )
                        
                        final_output = rfmt_all.copy().reset_index()
                        features = rfmt_all[['Recency', 'Frequency', 'Monetary', 'T']]
                        
                        # 2. Xử lý chuẩn hóa và gán phân cụm nhóm
                        features_scaled = scaler.transform(features)
                        final_output['Cluster_ID'] = kmeans.predict(features_scaled)
                        
                        # 3. Sử dụng mô hình XGBoost thích ứng dự báo kết quả đầu ra
                        all_predictions = predict_xgboost_smart(prediction_model, rfmt_all, features_scaled, final_output['Cluster_ID'].values)
                        final_output['Dự_Báo_Hành_Vi'] = np.where(all_predictions == 1, 'Sẽ tái mua (Will Return)', 'Nguy cơ rời bỏ (Churn)')
                        
                        st.success("🎉 Tiến trình xử lý hoàn tất! Kết quả phân khúc tổng hợp:")
                        st.dataframe(final_output, use_container_width=True)
                        
                        # Cung cấp nút tải tệp báo cáo hoàn chỉnh cho người dùng
                        st.download_button(
                            label="📥 Tải Xuất File Báo Cáo Phân Tích CRM (CSV)", 
                            data=final_output.to_csv(index=False).encode('utf-8'), 
                            file_name="crm_customer_analytics_report.csv", 
                            mime="text/csv"
                        )
                    except Exception as err:
                        st.error(f"❌ Có lỗi bất ngờ phát sinh trong quá trình tính toán hàng loạt: {err}")
else:
    st.info("👈 **Chào mừng bạn đến với hệ thống phân tích CRM! Vui lòng tải lên file dữ liệu giao dịch (.csv) từ bảng điều khiển bên trái để kích hoạt mô hình dự báo AI.**")
