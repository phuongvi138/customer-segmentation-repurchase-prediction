import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# 1. CẤU HÌNH GIAO DIỆN WEB DOCK
st.set_page_config(
    page_title="E-Commerce Customer Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Giao diện Tiêu đề (Khớp với nghiên cứu của bạn)
st.title("🛍️ Dual-Stage Customer Segmentation & Repurchase Prediction Framework")
st.markdown("""
Ứng dụng này triển khai mô hình **Hybrid Customer Segmentation & Prediction** được đề xuất trong bài báo nghiên cứu. 
Hệ thống tự động sử dụng bộ tính năng **RFMT** để nhận diện hành vi khách hàng và dự báo khả năng quay lại tái mua hàng (*Will Return*).
""")
st.write("---")

# 2. ĐƯỜNG DẪN ĐẾN FILE MODEL TRÊN GITHUB 
MODEL_PATH = "src/hybrid_customer_segmentation_prediction_model.pkl"

@st.cache_resource
def load_hybrid_model():
    try:
        # Tải mô hình đã đóng gói bằng joblib
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Không thể tải file model tại đường dẫn '{MODEL_PATH}'. Lỗi chi tiết: {e}")
        return None

model = load_hybrid_model()

# 3. CHIA TAB CHỨC NĂNG TRÊN GIAO DIỆN
tab1, tab2 = st.tabs(["🎯 Dự báo cho 1 Khách hàng", "📊 Xử lý hàng loạt (File CSV)"])

# --- TAB 1: DỰ BÁO CHO MỘT KHÁCH HÀNG ---
with tab1:
    st.header("Dự báo xu hướng tái mua hàng (Single Customer)")
    st.write("Điều chỉnh các chỉ số RFMT bên dưới để kiểm tra khả năng quay lại mua hàng của khách hàng.")

    # Tạo 4 cột để nhập các biến đầu vào bằng thanh trượt (Slider)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recency = st.slider("Recency (Số ngày kể từ lần mua cuối cùng)", min_value=0, max_value=365, value=30, step=1)
    with col2:
        frequency = st.slider("Frequency (Tổng số đơn hàng đã mua)", min_value=1, max_value=100, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary Value (Tổng số tiền đã chi tiêu £)", min_value=0.0, max_value=100000.0, value=250.0, step=10.0)
    with col4:
        time_t = st.slider("Time T (Nhịp điệu mua sắm - số ngày cách nhau giữa các đơn)", min_value=0, max_value=180, value=15, step=1)

    # Nút bấm kích hoạt dự báo
    if st.button("Phân tích khách hàng", type="primary"):
        if model is not None:
            # Tạo DataFrame từ các biến đầu vào theo đúng cấu trúc dữ liệu huấn luyện
            input_df = pd.DataFrame([[recency, frequency, monetary, time_t]], columns=['R', 'F', 'M', 'T'])
            
            try:
                # Thực hiện dự báo nhãn lớp (0 hoặc 1)
                prediction = model.predict(input_df)[0]
                
                # Thực hiện dự báo xác suất (nếu mô hình hỗ trợ predict_proba)
                try:
                    prob = model.predict_proba(input_df)[0][1]
                except:
                    # Phương án dự phòng nếu mô hình là dạng Hard-voting/không hỗ trợ xác suất
                    prob = 1.0 if prediction == 1 else 0.0

                # HIỂN THỊ KẾT QUẢ TRỰC QUAN
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
                        st.warning("Khuyến nghị quản trị: Kích hoạt ngay chiến dịch Marketing cá nhân hóa và mã giảm giá giữ chân khách hàng.")
            
            except Exception as e:
                st.error(f"❌ Lỗi khi thực hiện dự báo: {e}")
                st.info("💡 Lưu ý: Hãy đảm bảo file .pkl của bạn nhận đầu vào trực tiếp là 4 biến ['R', 'F', 'M', 'T'].")
        else:
            st.warning("Hệ thống chưa thể dự báo do thiếu file mô hình `.pkl`.")

# --- TAB 2: XỬ LÝ HÀNG LOẠT QUA FILE CSV ---
with tab2:
    st.header("Xử lý dữ liệu quy mô lớn (Batch Processing)")
    st.write("Tải lên file CSV chứa danh sách nhiều khách hàng (có sẵn các cột `R`, `F`, `M`, `T`) để chạy dự báo hàng loạt.")

    uploaded_file = st.file_uploader("Chọn file dữ liệu RFMT mẫu (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Kiểm tra xem file CSV tải lên có đủ các cột R, F, M, T hay không
        required_cols = {'R', 'F', 'M', 'T'}
        if not required_cols.issubset(data.columns):
            st.error("❌ Cấu trúc file không hợp lệ! File CSV bắt buộc phải chứa đủ 4 cột tiêu đề: R, F, M, T")
        else:
            st.success("✅ File được tải lên thành công!")
            
            if model is not None:
                try:
                    # Trích xuất các biến đặc trưng
                    features = data[['R', 'F', 'M', 'T']]
                    
                    # Dự báo hàng loạt cho toàn bộ file
                    data['Prediction_Label'] = model.predict(features)
                    data['Status'] = np.where(data['Prediction_Label'] == 1, 'Will Return', 'Not Return')
                    
                    # Hiển thị 10 dòng kết quả đầu tiên cho người dùng xem trước
                    st.subheader("Xem trước kết quả dự báo (10 dòng đầu tiên)")
                    st.dataframe(data.head(10), use_container_width=True)
                    
                    # Vẽ biểu đồ phân tích tỷ lệ khách hàng quay lại bằng Plotly
                    st.write("---")
                    st.subheader("Biểu đồ phân tích kết quả dự báo (Visual Analytics)")
                    
                    fig_pie = px.pie(data, names='Status', title='Tỷ lệ phân bổ hành vi Tái mua hàng tổng thể', 
                                     color='Status', color_discrete_map={'Will Return':'#2ca02c', 'Not Return':'#d62728'}, hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Tạo nút cho phép người dùng tải xuống file kết quả đã chấm điểm nhãn
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
