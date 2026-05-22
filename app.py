import warnings
warnings.filterwarnings("ignore")

import pickle
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0F0F13; color: #E8E6F0; }
[data-testid="stSidebar"] { background: #16151C !important; border-right: 1px solid #2A2835; }
[data-testid="stSidebar"] * { color: #C8C4D8 !important; }

.section-title { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; color: #6B6880; margin: 20px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #2A2835; }
.rfmt-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0; }
.rfmt-card { background: #1A1925; border: 1px solid #2A2835; border-radius: 12px; padding: 14px 16px; position: relative; overflow: hidden; }
.rfmt-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--accent); }
.rfmt-letter { font-family: 'DM Mono', monospace; font-size: 28px; font-weight: 500; color: var(--accent); line-height: 1; }
.rfmt-name { font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: #6B6880; margin-top: 2px; margin-bottom: 10px; }
.rfmt-value { font-size: 22px; font-weight: 600; color: #E8E6F0; line-height: 1; }
.rfmt-unit { font-size: 11px; color: #6B6880; margin-left: 3px; }

.seg-banner { background: #1A1925; border: 1px solid var(--seg-color); border-radius: 14px; padding: 18px 22px; display: flex; align-items: center; gap: 16px; margin: 14px 0; position: relative; overflow: hidden; }
.seg-banner::after { content: ''; position: absolute; inset: 0; background: var(--seg-color); opacity: .05; pointer-events: none; }
.seg-icon { font-size: 36px; }
.seg-label { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: #6B6880; }
.seg-name  { font-size: 22px; font-weight: 600; color: #E8E6F0; margin: 2px 0; }
.seg-meta  { font-size: 12px; color: #6B6880; }

.proba-row  { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.proba-label{ width: 170px; font-size: 12px; color: #C8C4D8; flex-shrink: 0; }
.proba-bar  { flex: 1; height: 6px; background: #2A2835; border-radius: 3px; overflow: hidden; }
.proba-fill { height: 100%; border-radius: 3px; transition: width .5s ease; }
.proba-pct  { width: 40px; text-align: right; font-size: 11px; font-family: 'DM Mono', monospace; color: #6B6880; }

.rec-box { background: #1A1925; border-left: 3px solid #7C6AF5; border-radius: 0 10px 10px 0; padding: 12px 16px; font-size: 13px; color: #C8C4D8; line-height: 1.7; margin-top: 10px; }
.rec-box strong { color: #E8E6F0; }

.pipeline { display: flex; align-items: center; gap: 0; margin: 16px 0; padding: 12px 16px; background: #1A1925; border-radius: 10px; }
.pip-step { display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 1; }
.pip-dot  { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 600; font-family: 'DM Mono', monospace; }
.pip-done   { background: #7C6AF5; color: #fff; }
.pip-active { background: transparent; border: 2px solid #7C6AF5; color: #7C6AF5; }
.pip-idle   { background: #2A2835; color: #6B6880; }
.pip-lbl    { font-size: 9px; letter-spacing: .06em; text-transform: uppercase; color: #6B6880; text-align: center; line-height: 1.3; }
.pip-line   { flex: 0 0 20px; height: 1px; background: #2A2835; }
.pip-line-done { background: #7C6AF5; }

.stButton > button { background: #7C6AF5 !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; padding: 10px 20px !important; }
.stButton > button:hover { background: #6A58E0 !important; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
hr { border-color: #2A2835 !important; }
[data-testid="stNumberInput"] input { background: #1A1925 !important; border: 1px solid #2A2835 !important; color: #E8E6F0 !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stExpander"] { background: #1A1925 !important; border: 1px solid #2A2835 !important; border-radius: 10px !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path="src/hybrid_customer_segmentation_prediction_model (1).pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file model pkl. Đặt file cùng thư mục với app.py")
    st.stop()

SCALER       = model["scaler"]
KMEANS       = model["kmeans_model"]
CLASSIFIERS  = model["segment_classifiers"]
CLUSTER_LBLS = model["cluster_labels"]
ANALYSIS_DT  = model["analysis_date"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
SEG_META = {
    "At-Risk / Dormant":    {"icon": "⚠️",  "color": "#E5854A", "action": "Gửi win-back campaign, ưu đãi giảm giá mạnh để kéo lại."},
    "Inactive / Churned":   {"icon": "💤",  "color": "#E24B4A", "action": "Thử re-engagement email lần cuối, sau đó archive."},
    "Potential Loyalists":  {"icon": "🌱",  "color": "#4CAF80", "action": "Mời tham gia loyalty program, tặng điểm double kỳ tới."},
    "Champions / VIPs":     {"icon": "👑",  "color": "#F5C842", "action": "Ưu tiên early access, tặng quà VIP, upsell gói premium."},
    "Occasional Buyers":    {"icon": "🛍️",  "color": "#7C6AF5", "action": "Gửi newsletter sản phẩm phù hợp, khuyến mãi theo mùa."},
}

BAR_COLORS = ["#7C6AF5", "#4CAF80", "#F5C842", "#E5854A", "#E24B4A"]

def compute_rfmt(txn_df: pd.DataFrame, analysis_date: pd.Timestamp):
    """Tính R, F, M, T và fix lỗi NaN khi parse Date"""
    df = txn_df.copy()
    
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    
    df["LineTotal"]   = df["Quantity"] * df["Price"]
    df = df[df["LineTotal"] > 0]
    
    if df.empty:
        raise ValueError("Dữ liệu trống sau khi lọc. Vui lòng kiểm tra lại định dạng Ngày (VD: 01.12.2009).")

    recency   = (analysis_date - df["InvoiceDate"].max()).days
    frequency = df["Invoice"].nunique()
    monetary  = df["LineTotal"].sum()

    dates = df.groupby("Invoice")["InvoiceDate"].min().sort_values()
    if len(dates) > 1:
        gaps = [(dates.iloc[i+1] - dates.iloc[i]).days for i in range(len(dates)-1)]
        t_val = float(np.mean(gaps))
    else:
        t_val = float(recency)

    return float(recency), float(frequency), float(monetary), float(t_val)

def predict(recency, frequency, monetary, t_val):
    X_rfmt   = pd.DataFrame([[recency, frequency, monetary, t_val]], columns=["Recency","Frequency","Monetary","T"])
    scaled   = SCALER.transform(X_rfmt)
    cluster  = int(KMEANS.predict(scaled)[0])
    seg_name = CLUSTER_LBLS.get(cluster, f"Cluster {cluster}")

    # FIX: Ép Cluster 1 (Inactive / Churned) luôn có tỉ lệ quay lại là 0%
    if cluster == 1:
        will_return_proba = 0.0
    else:
        clf = CLASSIFIERS.get(cluster)
        if clf:
            X_cls = np.array([[recency, frequency, monetary]])
            will_return_proba = float(clf.predict_proba(X_cls)[0][1])
        else:
            will_return_proba = None

    return cluster, seg_name, will_return_proba

# ─────────────────────────────────────────────
# SIDEBAR — INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;padding-bottom:14px;border-bottom:1px solid #2A2835;margin-bottom:16px;'>
            <span style='font-size:22px;'>🎯</span>
            <div>
                <div style='font-size:14px;font-weight:600;color:#E8E6F0;'>Customer Segmentation</div>
                <div style='font-size:10px;color:#6B6880;font-family:DM Mono;'>KMeans + XGBoost</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Thông tin khách hàng</div>', unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID", value="13085", label_visibility="collapsed")
    country     = st.text_input("Country", value="United Kingdom", label_visibility="collapsed")

    st.markdown('<div class="section-title">Lịch sử giao dịch</div>', unsafe_allow_html=True)
    st.caption("Định dạng Date: DD.MM.YYYY HH:MM")

    if "txn_data" not in st.session_state:
        st.session_state.txn_data = pd.DataFrame([
            {"Invoice": "489434", "InvoiceDate": "01.12.2009 07:45", "Quantity": 12, "Price": 6.95},
            {"Invoice": "489435", "InvoiceDate": "15.12.2009 09:30", "Quantity": 6,  "Price": 12.50},
            {"Invoice": "491203", "InvoiceDate": "03.01.2010 11:15", "Quantity": 4,  "Price": 8.75},
            {"Invoice": "493012", "InvoiceDate": "22.01.2010 14:00", "Quantity": 10, "Price": 5.40},
        ])

    # ĐÃ SỬA: Thêm thuộc tính hide_index=True để ẩn cột số thứ tự bên trái bảng
    edited_df = st.data_editor(
        st.session_state.txn_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Invoice":     st.column_config.TextColumn("Invoice", width="small"),
            "InvoiceDate": st.column_config.TextColumn("Date", width="medium"),
            "Quantity":    st.column_config.NumberColumn("Qty", min_value=1, width="small"),
            "Price":       st.column_config.NumberColumn("Price", min_value=0.01, format="%.2f", width="small"),
        },
        hide_index=True,
        key="txn_editor",
    )
    st.session_state.txn_data = edited_df

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn   = st.button("🔍  Tính RFMT & Phân loại", use_container_width=True)
    reset_btn = st.button("↺  Nhập lại", use_container_width=True)

    if reset_btn:
        st.session_state.txn_data = pd.DataFrame(columns=["Invoice","InvoiceDate","Quantity","Price"])
        if "result" in st.session_state: del st.session_state["result"]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;color:#3D3A50;font-family:DM Mono;line-height:1.8;'>analysis_date<br>{ANALYSIS_DT.strftime('%d %b %Y')}<br>k = {int(model['optimal_k'])} clusters</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
st.markdown("<h1 style='font-size:22px;font-weight:600;color:#E8E6F0;margin:0 0 4px;'>Customer Segmentation & Repurchase Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:13px;color:#6B6880;margin:0 0 20px;'>Nhập giao dịch thô → tính RFMT tự động → phân cụm → dự đoán khả năng quay lại</p>", unsafe_allow_html=True)

has_result = "result" in st.session_state

def pip_dot(label, num, state):
    cls = {"done":"pip-done","active":"pip-active","idle":"pip-idle"}[state]
    return f'<div class="pip-step"><div class="pip-dot {cls}">{num}</div><div class="pip-lbl">{label}</div></div>'

def pip_line(done=False):
    return f'<div class="pip-line {"pip-line-done" if done else ""}"></div>'

s = ["done"]*6 if has_result else ["active","idle","idle","idle","idle","idle"]

st.markdown(f"""
<div class="pipeline">
    {pip_dot("Nhập<br>giao dịch", "1", s[0])}{pip_line(s[1]=="done")}
    {pip_dot("Tính<br>RFMT", "2", s[1])}{pip_line(s[2]=="done")}
    {pip_dot("Scale<br>features", "3", s[2])}{pip_line(s[3]=="done")}
    {pip_dot("KMeans<br>cluster", "4", s[3])}{pip_line(s[4]=="done")}
    {pip_dot("XGBoost<br>score", "5", s[4])}{pip_line(s[5]=="done")}
    {pip_dot("Kết quả<br>& gợi ý", "6", s[5])}
</div>
""", unsafe_allow_html=True)

# ── RUN PREDICTION ───────────────────────────────────────────────
if run_btn:
    df = edited_df.copy()
    if df.empty or len(df) == 0:
        st.warning("Vui lòng nhập ít nhất 1 giao dịch.")
    else:
        try:
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
            df["Price"]    = pd.to_numeric(df["Price"],    errors="coerce").fillna(0)
            
            r, f, m, t = compute_rfmt(df, ANALYSIS_DT)
            cluster, seg_name, will_return = predict(r, f, m, t)

            # FIX: Cập nhật xác suất của tất cả các cụm trong bảng xếp hạng bên dưới luôn đồng bộ
            all_probas = {}
            for cid, lbl in CLUSTER_LBLS.items():
                if cid == 1:
                    p = 0.0
                else:
                    clf = CLASSIFIERS.get(cid)
                    if clf:
                        p = float(clf.predict_proba
