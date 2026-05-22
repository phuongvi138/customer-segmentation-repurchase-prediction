import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# 1. CẤU HÌNH GIAO GIỆN WEB DOCK
st.set_page_config(
    page_title="E-Commerce Customer Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Giao diện Tiêu đề
st.title("🛍️ Dual-Stage Customer Segmentation & Repurchase Prediction Framework")
st.markdown("""
Ứng dụng này triển khai mô hình **Hybrid Customer Segmentation & Prediction** được đề xuất trong bài báo nghiên cứu. 
Hệ thống tự động sử dụng bộ tính năng **RFMT** để nhận diện hành vi khách hàng và dự báo khả năng quay lại tái mua hàng (*Will Return*).
""")
st.write("---")

# 2. ĐƯỜNG DẪN ĐẾN FILE MODEL
MODEL_PATH = "src/hybrid_customer_segmentation_prediction_model (1).pkl"

@st.cache_resource
def load_hybrid_model():
    try:
        obj = joblib.load(MODEL_PATH)
        return obj
    except Exception as e:
        st.error(f"⚠️ Không thể tải file model tại đường dẫn '{MODEL_PATH}'. Lỗi chi tiết: {e}")
        return None

loaded_obj = load_hybrid_model()

# Xử lý bóc tách mô hình nếu file .pkl là Dictionary
model = None
if loaded_obj is not None:
    if isinstance(loaded_obj, dict):
        st.sidebar.info(f"🔍 Phát hiện file .pkl là Dictionary. Các khóa bên trong gồm: `{list(loaded_obj.keys())}`")
        
        # Thử tự động tìm mô hình dự báo từ các khóa phổ biến
        possible_keys = ['model', 'classification_model', 'classifier', 'xgb', 'xgboost', 'final_model', 'predict_model']
        for key in possible_keys:
            if key in loaded_obj:
                model = loaded_obj[key]
                st.sidebar.success(f"🎯 Đã tự động kích hoạt mô hình từ khóa: `{key}`")
                break
        
        # Nếu không khớp khóa nào ở trên, tự tìm thành phần nào có hàm predict
        if model == True:
            for k, v in loaded_obj.items():
                if hasattr(v, 'predict'):
                    model = v
                    st.sidebar.success(f"🎯 Đã tìm thấy mô hình khả dụng tại khóa: `{k}`")
                    break
        
        if model is None:
            st.error("❌ Không tìm thấy thành phần dự báo (Model) hợp lệ bên trong Dictionary của file .pkl. Vui lòng kiểm tra lại cấu trúc lưu file.")
    else:
        # Nếu file .pkl là một model thuần túy
        model = loaded_obj

# 3. CHIA TAB CHỨC NĂNG TRÊN GIAO DIỆN
tab1, tab2 = st.tabs(["🎯 Dự báo cho 1 Khách hàng", "📊 Xử lý hàng loạt (File CSV)"])

# --- TAB 1: DỰ BÁO CHO MỘT KHÁCH HÀNG ---
with tab1:
    st.header("Dự báo xu hướng tái mua hàng (Single Customer)")
    st.write("Điều chỉnh các chỉ số RFMT bên dưới để kiểm tra khả năng quay lại mua hàng của khách hàng.")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recency = st.slider("Recency (Số ngày kể từ lần mua cuối cùng)", min_value=0, max_value=365, value=30, step=1)
    with col2:
        frequency = st.slider("Frequency (Tổng số đơn hàng đã mua)", min_value=1, max_value=100, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary Value (Tổng số tiền đã chi tiêu £)", min_value=0.0, max_value=100000.0, value=250.0, step=10.0)
    with col4:
        time_t = st.slider("Time T (Nhịp điệu mua sắm - số ngày cách nhau giữa các đơn)", min_value=0, max_value=180, value=15, step=1)

    if st.button("Phân tích khách hàng", type="primary"):
        if model is not None:
            input_df = pd.DataFrame([[recency, frequency, monetary, time_t]], columns=['R', 'F', 'M', 'T'])
            
            try:
                # Thực hiện dự báo nhãn lớp (0 hoặc 1)
                prediction = model.predict(input_df)[0]
                
                # Thực hiện dự báo xác suất
                try:
                    prob = model.predict_proba(input_df)[0][1]
                except:
                    prob = 1.0 if prediction == 1 else 0.0

                st.write("---")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label="Trạng thái xử lý", value="Thành công (Success)")
                    st.write("Mô hình Hybrid tích hợp kỹ thuật cân bằng dữ liệu **SMOTE-Tomek** đã chấm điểm hồ sơ hành vi này.")
                    
                with res_col2:
                    if prediction == 1 or prob >= 0.5:
                        st.success(f"🔮 **DỰ BÁO: KHÁCH HÀNG SẼ QUAY LẠI (Will Return)** \n\nXác suất tái mua hàng: {prob*100:.2f}%")
                        st.balloons()
                    else:
                        st.error(f"🚨 **DỰ BÁO: NGUY CƠ RỜI BỎ / KHÔNG QUAY LẠI (Not Return)** \n\nXác suất rời đi: {(1-prob)*100:.2f}%")
                        st.warning("Khuyến nghị quản trị: Kích hoạt ngay chiến dịch Marketing cá nhân hóa để giữ chân khách hàng.")
            
            except Exception as e:
                st.error(f"❌ Lỗi khi thực hiện chạy hàm predict: {e}")
        else:
            st.warning("Hệ thống chưa có mô hình hợp lệ để dự báo. Vui lòng kiểm tra thanh SideBar bên trái để xem các Khóa (Keys) trong file.")

# --- TAB 2: XỬ LÝ HÀNG LOẠT QUA FILE CSV ---
with tab2:
    st.header("Xử lý dữ liệu quy mô lớn (Batch Processing)")
    st.write("Tải lên file CSV chứa danh sách nhiều khách hàng (có sẵn các cột `R`, `F`, `M`, `T`) để chạy dự báo hàng loạt.")

    uploaded_file = st.file_uploader("Chọn file dữ liệu RFMT mẫu (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        required_cols = {'R', 'F', 'M', 'T'}
        if not required_cols.issubset(data.columns):
            st.error("❌ Cấu trúc file không hợp lệ! File CSV bắt buộc phải chứa đủ 4 cột tiêu đề: R, F, M, T")
        else:
            st.success("✅ File được tải lên thành công!")
            
            if model is not None:
                try:
                    features = data[['R', 'F', 'M', 'T']]
                    data['Prediction_Label'] = model.predict(features)
                    data['Status'] = np.where(data['Prediction_Label'] == 1, 'Will Return', 'Not Return')
                    
                    st.subheader("Xem trước kết quả dự báo (10 dòng đầu tiên)")
                    st.dataframe(data.head(10), use_container_width=True)
                    
                    st.write("---")
                    st.subheader("Biểu đồ phân tích kết quả dự báo (Visual Analytics)")
                    
                    fig_pie = px.pie(data, names='Status', title='Tỷ lệ phân bổ hành vi Tái mua hàng tổng thể', 
                                     color='Status', color_discrete_map={'Will Return':'#2ca02c', 'Not Return':'#d62728'}, hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.write("---")
                    csv_data = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống file kết quả dự báo hoàn chỉnh (CSV)",
                        data=csv_data,
                        file_name="customer_repurchase_predictions_output.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi trong quá trình xử lý file hàng loạt: {e}")
