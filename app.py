"""
app.py — Customer Segmentation & Repurchase Prediction
Chạy: streamlit run app.py
Yêu cầu: pip install streamlit pandas numpy scikit-learn xgboost plotly
"""

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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main background ── */
.stApp { background: #0F0F13; color: #E8E6F0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #16151C !important;
    border-right: 1px solid #2A2835;
}
[data-testid="stSidebar"] * { color: #C8C4D8 !important; }

/* ── Section headers ── */
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #6B6880;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2A2835;
}

/* ── Metric cards ── */
.rfmt-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 12px 0;
}
.rfmt-card {
    background: #1A1925;
    border: 1px solid #2A2835;
    border-radius: 12px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.rfmt-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.rfmt-letter {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--accent);
    line-height: 1;
}
.rfmt-name {
    font-size: 10px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #6B6880;
    margin-top: 2px;
    margin-bottom: 10px;
}
.rfmt-value {
    font-size: 22px;
    font-weight: 600;
    color: #E8E6F0;
    line-height: 1;
}
.rfmt-unit {
    font-size: 11px;
    color: #6B6880;
    margin-left: 3px;
}

/* ── Segment banner ── */
.seg-banner {
    background: #1A1925;
    border: 1px solid var(--seg-color);
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 14px 0;
    position: relative;
    overflow: hidden;
}
.seg-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--seg-color);
    opacity: .05;
    pointer-events: none;
}
.seg-icon { font-size: 36px; }
.seg-label { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: #6B6880; }
.seg-name  { font-size: 22px; font-weight: 600; color: #E8E6F0; margin: 2px 0; }
.seg-meta  { font-size: 12px; color: #6B6880; }

/* ── Proba bar ── */
.proba-row  { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.proba-label{ width: 170px; font-size: 12px; color: #C8C4D8; flex-shrink: 0; }
.proba-bar  {
    flex: 1; height: 6px; background: #2A2835; border-radius: 3px; overflow: hidden;
}
.proba-fill { height: 100%; border-radius: 3px; transition: width .5s ease; }
.proba-pct  { width: 40px; text-align: right; font-size: 11px;
              font-family: 'DM Mono', monospace; color: #6B6880; }

/* ── Recommendation box ── */
.rec-box {
    background: #1A1925;
    border-left: 3px solid #7C6AF5;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #C8C4D8;
    line-height: 1.7;
    margin-top: 10px;
}
.rec-box strong { color: #E8E6F0; }

/* ── Pipeline steps ── */
.pipeline {
    display: flex; align-items: center; gap: 0; margin: 16px 0;
    padding: 12px 16px;
    background: #1A1925; border-radius: 10px;
}
.pip-step { display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 1; }
.pip-dot  {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; font-family: 'DM Mono', monospace;
}
.pip-done   { background: #7C6AF5; color: #fff; }
.pip-active { background: transparent; border: 2px solid #7C6AF5; color: #7C6AF5; }
.pip-idle   { background: #2A2835; color: #6B6880; }
.pip-lbl    { font-size: 9px; letter-spacing: .06em; text-transform: uppercase;
              color: #6B6880; text-align: center; line-height: 1.3; }
.pip-line   { flex: 0 0 20px; height: 1px; background: #2A2835; }
.pip-line-done { background: #7C6AF5; }

/* ── Transaction table ── */
.txn-df table { font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: #7C6AF5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.stButton > button:hover { background: #6A58E0 !important; }

/* ── Data editor ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #2A2835 !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: #1A1925 !important;
    border: 1px solid #2A2835 !important;
    color: #E8E6F0 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1A1925 !important;
    border: 1px solid #2A2835 !important;
    border-radius: 10px !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path="hybrid_customer_segmentation_prediction_model (1).pkl"):
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
    """Tính R, F, M, T từ bảng giao dịch thô."""
    df = txn_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["LineTotal"]   = df["Quantity"] * df["Price"]
    df = df[df["LineTotal"] > 0]

    recency   = (analysis_date - df["InvoiceDate"].max()).days
    frequency = df["Invoice"].nunique()
    monetary  = df["LineTotal"].sum()

    dates = df.groupby("Invoice")["InvoiceDate"].min().sort_values()
    if len(dates) > 1:
        gaps = [(dates.iloc[i+1] - dates.iloc[i]).days for i in range(len(dates)-1)]
        t_val = float(np.mean(gaps))
    else:
        t_val = float(recency)

    return recency, frequency, monetary, t_val


def predict(recency, frequency, monetary, t_val):
    """Scale → KMeans cluster → XGB proba."""
    X_rfmt   = pd.DataFrame([[recency, frequency, monetary, t_val]],
                            columns=["Recency","Frequency","Monetary","T"])
    scaled   = SCALER.transform(X_rfmt)
    cluster  = int(KMEANS.predict(scaled)[0])
    seg_name = CLUSTER_LBLS.get(cluster, f"Cluster {cluster}")

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
        <div style='display:flex;align-items:center;gap:10px;padding-bottom:14px;
                    border-bottom:1px solid #2A2835;margin-bottom:16px;'>
            <span style='font-size:22px;'>🎯</span>
            <div>
                <div style='font-size:14px;font-weight:600;color:#E8E6F0;'>Customer Segmentation</div>
                <div style='font-size:10px;color:#6B6880;font-family:DM Mono;'>KMeans + XGBoost</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Thông tin khách hàng</div>', unsafe_allow_html=True)
    customer_id = st.text_input("Customer ID", value="13085", label_visibility="collapsed",
                                placeholder="Customer ID")
    country     = st.text_input("Country", value="United Kingdom", label_visibility="collapsed",
                                placeholder="Country")

    st.markdown('<div class="section-title">Lịch sử giao dịch</div>', unsafe_allow_html=True)
    st.caption("Nhập từng dòng giao dịch — Invoice, Date (DD.MM.YYYY HH:MM), Qty, Price")

    # Default transactions
    if "txn_data" not in st.session_state:
        st.session_state.txn_data = pd.DataFrame([
            {"Invoice": "489434", "InvoiceDate": "01.12.2009 07:45", "Quantity": 12, "Price": 6.95},
            {"Invoice": "489435", "InvoiceDate": "15.12.2009 09:30", "Quantity": 6,  "Price": 12.50},
            {"Invoice": "491203", "InvoiceDate": "03.01.2010 11:15", "Quantity": 4,  "Price": 8.75},
            {"Invoice": "493012", "InvoiceDate": "22.01.2010 14:00", "Quantity": 10, "Price": 5.40},
        ])

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
        st.session_state.txn_data = pd.DataFrame(
            columns=["Invoice","InvoiceDate","Quantity","Price"]
        )
        if "result" in st.session_state:
            del st.session_state["result"]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='font-size:10px;color:#3D3A50;font-family:DM Mono;line-height:1.8;'>
            analysis_date<br>{ANALYSIS_DT.strftime('%d %b %Y')}<br>
            k = {int(model['optimal_k'])} clusters
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='font-size:22px;font-weight:600;color:#E8E6F0;margin:0 0 4px;'>
        Customer Segmentation & Repurchase Prediction
    </h1>
    <p style='font-size:13px;color:#6B6880;margin:0 0 20px;'>
        Nhập giao dịch thô → tính RFMT tự động → phân cụm → dự đoán khả năng quay lại
    </p>
""", unsafe_allow_html=True)

# ── PIPELINE STATUS ──────────────────────────────────────────────
has_result = "result" in st.session_state

def pip_dot(label, num, state):
    cls = {"done":"pip-done","active":"pip-active","idle":"pip-idle"}[state]
    return f"""
    <div class="pip-step">
        <div class="pip-dot {cls}">{num}</div>
        <div class="pip-lbl">{label}</div>
    </div>"""

def pip_line(done=False):
    cls = "pip-line-done" if done else ""
    return f'<div class="pip-line {cls}"></div>'

if not has_result:
    s = ["active","idle","idle","idle","idle","idle"]
else:
    s = ["done","done","done","done","done","done"]

st.markdown(f"""
<div class="pipeline">
    {pip_dot("Nhập<br>giao dịch", "1", s[0])}
    {pip_line(s[1]=="done")}
    {pip_dot("Tính<br>RFMT", "2", s[1])}
    {pip_line(s[2]=="done")}
    {pip_dot("Scale<br>features", "3", s[2])}
    {pip_line(s[3]=="done")}
    {pip_dot("KMeans<br>cluster", "4", s[3])}
    {pip_line(s[4]=="done")}
    {pip_dot("XGBoost<br>score", "5", s[4])}
    {pip_line(s[5]=="done")}
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
            df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

            r, f, m, t = compute_rfmt(df, ANALYSIS_DT)
            cluster, seg_name, will_return = predict(r, f, m, t)

            # Gather all cluster probas for display
            all_probas = {}
            for cid, clf in CLASSIFIERS.items():
                p = float(clf.predict_proba(np.array([[r, f, m]]))[0][1])
                all_probas[CLUSTER_LBLS[cid]] = p

            st.session_state["result"] = {
                "r": r, "f": f, "m": m, "t": t,
                "cluster": cluster, "seg_name": seg_name,
                "will_return": will_return,
                "all_probas": all_probas,
                "customer_id": customer_id, "country": country,
            }
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi xử lý: {e}")


# ── DISPLAY RESULTS ──────────────────────────────────────────────
if has_result:
    res = st.session_state["result"]
    r, f, m, t   = res["r"], res["f"], res["m"], res["t"]
    seg_name     = res["seg_name"]
    will_return  = res["will_return"]
    meta         = SEG_META.get(seg_name, {"icon":"❓","color":"#7C6AF5","action":"Không có gợi ý."})

    # ── RFMT Cards ────────────────────────────────
    st.markdown('<div class="section-title">Chỉ số RFMT được tính tự động</div>', unsafe_allow_html=True)

    rfmt_items = [
        ("R", "Recency",  f"{int(r)}",      "ngày",  "#7C6AF5"),
        ("F", "Frequency",f"{int(f)}",      "đơn",   "#4CAF80"),
        ("M", "Monetary", f"{m:,.0f}",      "đ",     "#F5C842"),
        ("T", "Avg interval", f"{t:.1f}",   "ngày/đơn","#E5854A"),
    ]

    cols = st.columns(4)
    for col, (letter, name, val, unit, accent) in zip(cols, rfmt_items):
        with col:
            st.markdown(f"""
            <div class="rfmt-card" style="--accent:{accent}">
                <div class="rfmt-letter">{letter}</div>
                <div class="rfmt-name">{name}</div>
                <div class="rfmt-value">{val}<span class="rfmt-unit">{unit}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: Segment banner + Will Return ──
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">Segment</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="seg-banner" style="--seg-color:{meta['color']}">
            <div class="seg-icon">{meta['icon']}</div>
            <div>
                <div class="seg-label">Cluster #{res['cluster']} · {int(model['optimal_k'])} clusters</div>
                <div class="seg-name">{seg_name}</div>
                <div class="seg-meta">Customer {res['customer_id']} · {res['country']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Khả năng quay lại</div>', unsafe_allow_html=True)
        if will_return is not None:
            pct = will_return * 100
            color = "#4CAF80" if pct >= 60 else "#E5854A" if pct >= 35 else "#E24B4A"
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = pct,
                number= {"suffix": "%", "font": {"size": 32, "color": "#E8E6F0",
                                                  "family": "DM Mono"}},
                gauge = {
                    "axis"     : {"range": [0, 100], "tickfont": {"color": "#6B6880", "size": 10}},
                    "bar"      : {"color": color, "thickness": 0.22},
                    "bgcolor"  : "#1A1925",
                    "borderwidth": 0,
                    "steps"    : [
                        {"range": [0,  35], "color": "#1F1E28"},
                        {"range": [35, 60], "color": "#201E2A"},
                        {"range": [60,100], "color": "#1E2026"},
                    ],
                    "threshold": {"line": {"color": color, "width": 2},
                                  "thickness": 0.8, "value": pct},
                },
            ))
            fig.update_layout(
                height=180, margin=dict(t=20, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8E6F0",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Cluster này không đủ dữ liệu để dự đoán will_return.")

    # ── Proba bars per cluster ─────────────────────
    if res["all_probas"]:
        st.markdown('<div class="section-title">Will_Return theo từng cluster</div>', unsafe_allow_html=True)
        sorted_probas = sorted(res["all_probas"].items(), key=lambda x: x[1], reverse=True)
        html_bars = ""
        for i, (lbl, prob) in enumerate(sorted_probas):
            pct   = prob * 100
            clr   = BAR_COLORS[i % len(BAR_COLORS)]
            is_current = "★ " if lbl == seg_name else ""
            html_bars += f"""
            <div class="proba-row">
                <div class="proba-label">{is_current}{lbl}</div>
                <div class="proba-bar">
                    <div class="proba-fill" style="width:{pct:.1f}%;background:{clr};"></div>
                </div>
                <div class="proba-pct">{pct:.1f}%</div>
            </div>"""
        st.markdown(html_bars, unsafe_allow_html=True)

    # ── Recommendation ────────────────────────────
    st.markdown(f"""
    <div class="section-title">Gợi ý hành động</div>
    <div class="rec-box">
        <strong>{meta['icon']} {seg_name}</strong> — {meta['action']}
    </div>
    """, unsafe_allow_html=True)

    # ── Debug expander ────────────────────────────
    with st.expander("🔧 Chi tiết kỹ thuật"):
        st.markdown(f"""
        <div style='font-family:DM Mono;font-size:11px;color:#6B6880;line-height:2;'>
            Cluster ID     : {res['cluster']}<br>
            Scaled features: R={r:.1f} F={f:.1f} M={m:.1f} T={t:.1f}<br>
            Analysis date  : {ANALYSIS_DT.strftime('%Y-%m-%d')}<br>
            Classifier     : XGBClassifier (binary:logistic)<br>
            Features used  : Recency, Frequency, Monetary_Past
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:48px;margin-bottom:16px;'>📊</div>
        <div style='font-size:16px;font-weight:500;color:#6B6880;'>
            Nhập giao dịch bên trái và nhấn <strong style="color:#7C6AF5;">Tính RFMT & Phân loại</strong>
        </div>
        <div style='font-size:12px;color:#3D3A50;margin-top:8px;'>
            App sẽ tự động tính R, F, M, T → phân cụm KMeans → dự đoán khả năng quay lại
        </div>
    </div>
    """, unsafe_allow_html=True)
