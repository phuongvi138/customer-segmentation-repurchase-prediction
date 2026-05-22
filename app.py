import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="CRM Analytics & Prediction", page_icon="🛍️", layout="wide")

st.title("🛍️ Hệ thống Tra cứu CRM & Dự báo Hành vi Khách hàng")
st.write("---")

# ==========================================
# 2. TẢI VÀ BÓC TÁCH FILE .PKL
# ==========================================
MODEL_PATH = "src/hybrid_customer_segmentation_prediction_model (1).pkl"

@st.cache_resource
def load_models():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

pkl_obj = load_models()
scaler, kmeans, prediction_model = None, None, None

if isinstance(pkl_obj, dict):
    for k, v in pkl_obj.items():
        k_lower = k.lower()
        if 'scale' in k_lower: scaler = v
        elif 'mean' in k_lower or 'cluster' in k_lower: kmeans = v
        elif 'model' in k_lower or 'xgb' in k_lower or 'class' in k_lower or 'pred' in k_lower: prediction_model = v

# ==========================================
# 3. KẾT NỐI CƠ SỞ DỮ LIỆU (DATABASE)
# ==========================================
st.sidebar.header("📂 1. Nạp Cơ sở dữ liệu")
st.sidebar.write("Tải file lịch sử giao dịch tổng (Raw Data) lên đây để hệ thống làm cơ sở tra cứu.")
db_file = st.sidebar.file_uploader("Tải DB (.csv)", type=["csv"])

# ==========================================
# 4. HÀM XỬ LÝ LỖI 10 BIẾN (ONE-HOT ENCODING)
# ==========================================
def prepare_10_features(rfmt_df, cluster_id, xgb_model):
    """Hàm này biến 4 biến RFMT + 1 Cluster ID thành 10 biến cho XGBoost"""
    # Lấy tên 10 cột chuẩn mà XGBoost đang đòi hỏi (từ ảnh của bạn)
    try:
        expected_cols = xgb_model.feature_names_in_
    except:
        try:
            expected_cols = xgb_model.get_booster().feature_names
        except:
            # Fallback mặc định nếu không tự lấy được tên cột từ model
            expected_cols = ['Recency', 'Frequency', 'Monetary', 'T', 'Cluster_0', 'Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5']
    
    # Tạo DataFrame chứa toàn bộ số 0 với 10 cột chuẩn
    df_10 = pd.DataFrame(0, index=[0], columns=expected_cols)
    
    # Điền 4 biến RFMT vào
    df_10['Recency'] = rfmt_df['Recency'].values[0]
    df_10['Frequency'] = rfmt_df['Frequency'].values[0]
    df_10['Monetary'] = rfmt_df['Monetary'].values[0]
    df_10['T'] = rfmt_df['T'].values[0]
    
    # Bật số 1 cho đúng cái Cluster_ID tương ứng (One-Hot Encoding)
    cluster_col_name = f'Cluster_{cluster_id}'
    if cluster_col_name in df_10.columns:
        df_10[cluster_col_name] = 1
        
    return df_10

# ==========================================
# 5. GIAO DIỆN TRA CỨU TỪNG KHÁCH HÀNG (CRM)
# ==========================================
if db_file is not None:
    # Đọc Database
    db_df = pd.read_csv(db_file)
    db_df['InvoiceDate'] = pd.to_datetime(db_df['InvoiceDate'])
    
    st.header("🔍 2. Tra cứu Khách hàng (CRM)")
    search_id = st.text_input("👤 Nhập Mã Khách Hàng (CustomerID) cần tra cứu:", placeholder="VD: 12345")
    
    if search_id:
        # Lọc dữ liệu của khách hàng này
        try:
            # Xử lý trường hợp ID là số hay chuỗi
            if db_df['CustomerID'].dtype == 'int64' or db_df['CustomerID'].dtype == 'float64':
                search_id_clean = float(search_id)
            else:
                search_id_clean = str(search_id)
                
            user_history = db_df[db_df['CustomerID'] == search_id_clean]
            
            if user_history.empty:
                st.warning(f"⚠️ Không tìm thấy khách hàng nào mang mã '{search_id}' trong cơ sở dữ liệu.")
            else:
                st.success(f"✅ Đã tìm thấy khách hàng {search_id}!")
                
                # --- PHẦN A: HIỂN THỊ LỊCH SỬ MUA HÀNG ---
                st.subheader("🛒 Lịch sử Giao dịch (Transaction History)")
                st.dataframe(user_history.sort_values(by='InvoiceDate', ascending=False), use_container_width=True)
                
                # --- PHẦN B: TÍNH TOÁN RFMT ---
                max_date_db = db_df['InvoiceDate'].max() # Lấy mốc thời gian chung của cả DB
                user_max_date = user_history['InvoiceDate'].max()
                user_min_date = user_history['InvoiceDate'].min()
                
                r_val = (max_date_db - user_max_date).days
                f_val = len(user_history)
                m_val = user_history['Total_Amount'].sum()
                t_val = (user_max_date - user_min_date).days
                
                st.subheader("📊 Chỉ số Hành vi (RFMT Metrics)")
                cols = st.columns(4)
                cols[0].metric("Recency (R)", f"{r_val} ngày")
                cols[1].metric("Frequency (F)", f"{f_val} đơn")
                cols[2].metric("Monetary (M)", f"£{m_val:,.2f}")
                cols[3].metric("Time T (T)", f"{t_val} ngày")
                
                # --- PHẦN C: DỰ BÁO BẰNG MÔ HÌNH HYBRID ---
                if None not in [scaler, kmeans, prediction_model]:
                    # 1. Chuẩn bị 4 biến cho K-Means
                    input_4 = pd.DataFrame([[r_val, f_val, m_val, t_val]], columns=['Recency', 'Frequency', 'Monetary', 'T'])
                    scaled_4 = scaler.transform(input_4)
                    
                    # 2. Phân cụm
                    cluster_id = kmeans.predict(scaled_4)[0]
                    
                    # 3. Chuẩn bị 10 biến cho XGBoost
                    input_10 = prepare_10_features(input_4, cluster_id, prediction_model)
                    
                    # 4. Dự báo
                    pred = prediction_model.predict(input_10)[0]
                    
                    st.subheader("🤖 Kết quả Phân tích từ AI")
                    c1, c2 = st.columns(2)
                    c1.info(f"🏷️ **Phân khúc:** Khách hàng thuộc **Nhóm (Cluster) {cluster_id}**")
                    
                    if pred == 1:
                        c2.success("🔮 **Dự báo 30 ngày tới:** KHÁCH HÀNG SẼ QUAY LẠI TÁI MUA (Will Return) 🎉")
                    else:
                        c2.error("🚨 **Dự báo 30 ngày tới:** KHÔNG QUAY LẠI (Nguy cơ rời bỏ/Churn) 📉")
                else:
                    st.error("Lỗi: Không tải được đầy đủ mô hình từ file .pkl.")
                    
        except ValueError:
            st.error("Vui lòng nhập CustomerID hợp lệ.")
else:
    st.info("👈 Vui lòng tải file Cơ sở dữ liệu (Database) lịch sử giao dịch ở menu bên trái để bắt đầu tra cứu.")
