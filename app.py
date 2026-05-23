"""
app.py — Customer Segmentation Streamlit App (fixed)
"""
import warnings
warnings.filterwarnings("ignore")

import pickle, datetime as dt, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0F0F13; color: #E8E6F0; }
[data-testid="stSidebar"] { background: #16151C !important; border-right: 1px solid #2A2835; }
[data-testid="stSidebar"] * { color: #C8C4D8 !important; }

.rfmt-card {
    background: #1A1925; border: 1px solid #2A2835; border-radius: 12px;
    padding: 14px 16px; position: relative; overflow: hidden; margin-bottom: 4px;
}
.rfmt-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent);
}
.rfmt-letter { font-family: 'DM Mono', monospace; font-size: 28px; font-weight: 500; color: var(--accent); line-height: 1; }
.rfmt-name   { font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: #6B6880; margin: 2px 0 10px; }
.rfmt-value  { font-size: 22px; font-weight: 600; color: #E8E6F0; }
.rfmt-unit   { font-size: 11px; color: #6B6880; margin-left: 3px; }

.seg-banner {
    background: #1A1925; border-radius: 14px; padding: 18px 22px;
    display: flex; align-items: center; gap: 16px; margin: 10px 0;
    border: 1px solid var(--seg-color); position: relative; overflow: hidden;
}
.seg-banner::after {
    content:''; position:absolute; inset:0;
    background:var(--seg-color); opacity:.05; pointer-events:none;
}
.seg-icon  { font-size: 36px; }
.seg-lbl   { font-size: 10px; letter-spacing:.12em; text-transform:uppercase; color:#6B6880; }
.seg-name  { font-size: 22px; font-weight: 600; color: #E8E6F0; margin: 2px 0; }
.seg-meta  { font-size: 12px; color: #6B6880; }

.proba-row   { display:flex; align-items:center; gap:10px; margin:5px 0; }
.proba-label { width:170px; font-size:12px; color:#C8C4D8; flex-shrink:0; }
.proba-bg    { flex:1; height:6px; background:#2A2835; border-radius:3px; overflow:hidden; }
.proba-fill  { height:100%; border-radius:3px; }
.proba-pct   { width:40px; text-align:right; font-size:11px; font-family:'DM Mono',monospace; color:#6B6880; }

.rec-box {
    background:#1A1925; border-left:3px solid #7C6AF5;
    border-radius:0 10px 10px 0; padding:12px 16px;
    font-size:13px; color:#C8C4D8; line-height:1.7; margin-top:10px;
}
.rec-box strong { color:#E8E6F0; }

.sec-title {
    font-family:'DM Mono',monospace; font-size:10px; letter-spacing:.14em;
    text-transform:uppercase; color:#6B6880; margin:20px 0 10px;
    padding-bottom:6px; border-bottom:1px solid #2A2835;
}

.stButton > button {
    background: #7C6AF5 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover { background: #6A58E0 !important; }

[data-testid="stNumberInput"] input {
    background:#1A1925 !important; border:1px solid #2A2835 !important;
    color:#E8E6F0 !important; border-radius:8px !important;
}
[data-testid="stExpander"] {
    background:#1A1925 !important; border:1px solid #2A2835 !important; border-radius:10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "src/hybrid_customer_segmentation_prediction_model (1).pkl"
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

SEG_META = {
    "At-Risk / Dormant":   {"icon":"⚠️",  "color":"#E5854A", "action":"Gửi win-back campaign, ưu đãi giảm giá mạnh để kéo lại."},
    "Inactive / Churned":  {"icon":"💤",  "color":"#E24B4A", "action":"Thử re-engagement email lần cuối, sau đó archive."},
    "Potential Loyalists": {"icon":"🌱",  "color":"#4CAF80", "action":"Mời tham gia loyalty program, tặng điểm double kỳ tới."},
    "Champions / VIPs":    {"icon":"👑",  "color":"#F5C842", "action":"Ưu tiên early access, tặng quà VIP, upsell gói premium."},
    "Occasional Buyers":   {"icon":"🛍️", "color":"#7C6AF5", "action":"Gửi newsletter sản phẩm phù hợp, khuyến mãi theo mùa."},
}
BAR_COLORS = ["#7C6AF5","#4CAF80","#F5C842","#E5854A","#E24B4A"]

# ─── HELPERS ─────────────────────────────────────────────────────
def compute_rfmt(txn_df, analysis_date):
    df = txn_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Price"]    = pd.to_numeric(df["Price"],    errors="coerce").fillna(0)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    if df.empty:
        return None, None, None, None
    df["LineTotal"] = df["Quantity"] * df["Price"]
    recency   = (analysis_date - df["InvoiceDate"].max()).days
    frequency = df["Invoice"].nunique()
    monetary  = df["LineTotal"].sum()
    dates = df.groupby("Invoice")["InvoiceDate"].min().sort_values()
    if len(dates) > 1:
        gaps  = [(dates.iloc[i+1] - dates.iloc[i]).days for i in range(len(dates)-1)]
        t_val = float(np.mean(gaps))
    else:
        t_val = float(recency)
    return recency, frequency, monetary, t_val

def predict(r, f, m, t):
    X_rfmt  = pd.DataFrame([[r, f, m, t]], columns=["Recency","Frequency","Monetary","T"])
    scaled  = SCALER.transform(X_rfmt)
    cluster = int(KMEANS.predict(scaled)[0])
    seg     = CLUSTER_LBLS.get(cluster, f"Cluster {cluster}")
    clf     = CLASSIFIERS.get(cluster)
    wr      = float(clf.predict_proba(np.array([[r, f, m]]))[0][1]) if clf else None
    all_p   = {}
    for cid, c in CLASSIFIERS.items():
        all_p[CLUSTER_LBLS[cid]] = float(c.predict_proba(np.array([[r, f, m]]))[0][1])
    return cluster, seg, wr, all_p

# ─── SESSION STATE init ──────────────────────────────────────────
DEFAULT_TXN = [
    {"Invoice":"489434","InvoiceDate":"01.12.2009 07:45","Quantity":12,"Price":6.95},
    {"Invoice":"489435","InvoiceDate":"15.12.2009 09:30","Quantity":6, "Price":12.50},
    {"Invoice":"491203","InvoiceDate":"03.01.2010 11:15","Quantity":4, "Price":8.75},
    {"Invoice":"493012","InvoiceDate":"22.01.2010 14:00","Quantity":10,"Price":5.40},
]
if "txn_data" not in st.session_state:
    # Reset index để tránh cột index thừa hiện ra trong data_editor
    st.session_state.txn_data = pd.DataFrame(DEFAULT_TXN).reset_index(drop=True)
if "result"   not in st.session_state:
    st.session_state.result = None
if "cust_id"  not in st.session_state:
    st.session_state.cust_id = "13085"
if "country"  not in st.session_state:
    st.session_state.country = "United Kingdom"

# ─── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding-bottom:14px;
                border-bottom:1px solid #2A2835;margin-bottom:16px;'>
        <span style='font-size:22px;'>🎯</span>
        <div>
            <div style='font-size:14px;font-weight:600;color:#E8E6F0;'>Customer Segmentation</div>
            <div style='font-size:10px;color:#6B6880;font-family:DM Mono,monospace;'>KMeans + XGBoost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#6B6880;margin-bottom:6px;">Thông tin khách hàng</p>', unsafe_allow_html=True)

    # Dùng key cố định, KHÔNG rerun khi thay đổi
    cust_id = st.text_input("Customer ID", value=st.session_state.cust_id,
                             key="input_cust_id", label_visibility="collapsed",
                             placeholder="Customer ID")
    country = st.text_input("Country", value=st.session_state.country,
                             key="input_country", label_visibility="collapsed",
                             placeholder="Country")

    st.markdown('<p style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#6B6880;margin:14px 0 4px;">Lịch sử giao dịch</p>', unsafe_allow_html=True)
    st.caption("Thêm/xoá dòng thoải mái — dữ liệu được giữ tự động")

    # on_change callback: lưu ngay khi người dùng edit xong 1 ô
    def _save_edits():
        # Streamlit lưu state của data_editor vào key "txn_editor"
        # Dạng {"edited_rows":{}, "added_rows":[], "deleted_rows":[]}
        changes = st.session_state.get("txn_editor", {})
        df = st.session_state.txn_data.copy()

        # Apply deleted rows (xử lý ngược để index không bị lệch)
        for idx in sorted(changes.get("deleted_rows", []), reverse=True):
            if idx < len(df):
                df = df.drop(df.index[idx]).reset_index(drop=True)

        # Apply edited rows
        for idx_str, edits in changes.get("edited_rows", {}).items():
            idx = int(idx_str)
            if idx < len(df):
                for col, val in edits.items():
                    df.at[idx, col] = val

        # Apply added rows
        for row in changes.get("added_rows", []):
            new_row = {"Invoice": "", "InvoiceDate": "", "Quantity": 1, "Price": 0.01}
            new_row.update(row)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        st.session_state.txn_data = df.reset_index(drop=True)

    st.data_editor(
        st.session_state.txn_data,
        num_rows="dynamic",
        use_container_width=True,
        key="txn_editor",
        on_change=_save_edits,
        column_config={
            "Invoice":     st.column_config.TextColumn("Mã đơn hàng", width="small"),
            "InvoiceDate": st.column_config.TextColumn("Ngày (DD.MM.YYYY HH:MM)", width="medium"),
            "Quantity":    st.column_config.NumberColumn("Số lượng", min_value=1,    width="small"),
            "Price":       st.column_config.NumberColumn("Đơn giá",  min_value=0.01, format="%.2f", width="small"),
        },
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        run_btn   = st.button("🔍 Phân loại", use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Nhập lại",  use_container_width=True)

    st.markdown(f"""
    <div style='margin-top:20px;font-size:10px;color:#3D3A50;font-family:DM Mono,monospace;line-height:1.9;'>
        analysis_date<br>{ANALYSIS_DT.strftime('%d %b %Y')}<br>
        k = {int(model['optimal_k'])} clusters
    </div>
    """, unsafe_allow_html=True)

# ─── BUTTON ACTIONS (không dùng st.rerun()) ──────────────────────
if reset_btn:
    st.session_state.txn_data = pd.DataFrame(DEFAULT_TXN)
    st.session_state.result   = None
    st.rerun()

if run_btn:
    st.session_state.cust_id = cust_id
    st.session_state.country = country
    df_run = st.session_state.txn_data.copy()

    if df_run.empty:
        st.session_state.result = {"error": "Vui lòng nhap it nhat 1 giao dich."}
    else:
        try:
            df_run["Quantity"] = pd.to_numeric(df_run["Quantity"], errors="coerce").fillna(0)
            df_run["Price"]    = pd.to_numeric(df_run["Price"],    errors="coerce").fillna(0)
            r, f, m, t = compute_rfmt(df_run, ANALYSIS_DT)
            if r is None:
                st.session_state.result = {"error": "Du lieu khong hop le."}
            else:
                cluster, seg, wr, all_p = predict(r, f, m, t)
                st.session_state.result = {
                    "r":r, "f":f, "m":m, "t":t,
                    "cluster":cluster, "seg":seg,
                    "will_return":wr, "all_probas":all_p,
                    "cust_id":cust_id, "country":country,
                }
        except Exception as e:
            st.session_state.result = {"error": f"Loi: {e}"}
    st.rerun()

# ─── MAIN PANEL ──────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:22px;font-weight:600;color:#E8E6F0;margin:0 0 4px;'>
    Customer Segmentation & Repurchase Prediction
</h1>
<p style='font-size:13px;color:#6B6880;margin:0 0 20px;'>
    Nhập giao dịch thô → tính RFMT tự động → phân cụm → dự đoán khả năng quay lại
</p>
""", unsafe_allow_html=True)

res = st.session_state.result

# ── Pipeline (dùng st.columns thay HTML) ─────────────────────────
STEPS = ["Nhập\ngiao dịch","Tính\nRFMT","Scale\nfeatures","KMeans\ncluster","XGBoost\nscore","Kết quả\n& gợi ý"]
done  = res is not None and "error" not in res

pip_cols = st.columns(len(STEPS) * 2 - 1)
for i, label in enumerate(STEPS):
    col_idx = i * 2
    with pip_cols[col_idx]:
        if done:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='width:30px;height:30px;border-radius:50%;background:#7C6AF5;
                            color:#fff;display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:600;margin:0 auto 4px;
                            font-family:DM Mono,monospace;'>{i+1}</div>
                <div style='font-size:9px;letter-spacing:.05em;text-transform:uppercase;
                            color:#7C6AF5;text-align:center;line-height:1.3;white-space:pre-line;'>{label}</div>
            </div>""", unsafe_allow_html=True)
        elif i == 0:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='width:30px;height:30px;border-radius:50%;background:transparent;
                            border:2px solid #7C6AF5;color:#7C6AF5;
                            display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:600;margin:0 auto 4px;
                            font-family:DM Mono,monospace;'>{i+1}</div>
                <div style='font-size:9px;letter-spacing:.05em;text-transform:uppercase;
                            color:#7C6AF5;text-align:center;line-height:1.3;white-space:pre-line;'>{label}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='width:30px;height:30px;border-radius:50%;background:#2A2835;
                            color:#6B6880;display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:600;margin:0 auto 4px;
                            font-family:DM Mono,monospace;'>{i+1}</div>
                <div style='font-size:9px;letter-spacing:.05em;text-transform:uppercase;
                            color:#6B6880;text-align:center;line-height:1.3;white-space:pre-line;'>{label}</div>
            </div>""", unsafe_allow_html=True)
    # line giữa các bước
    if i < len(STEPS) - 1:
        with pip_cols[col_idx + 1]:
            color = "#7C6AF5" if done else "#2A2835"
            st.markdown(f"<div style='height:1px;background:{color};margin-top:15px;'></div>",
                        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Error ─────────────────────────────────────────────────────────
if res and "error" in res:
    st.error(res["error"])

# ── Results ───────────────────────────────────────────────────────
elif res:
    r, f, m, t  = res["r"], res["f"], res["m"], res["t"]
    seg         = res["seg"]
    will_return = res["will_return"]
    all_probas  = res["all_probas"]
    meta        = SEG_META.get(seg, {"icon":"❓","color":"#7C6AF5","action":"Không có gợi ý."})

    # RFMT cards
    st.markdown('<div class="sec-title">Chỉ số RFMT được tính tự động</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"R","Recency",     f"{int(r)}",   "ngày",     "#7C6AF5"),
        (c2,"F","Frequency",   f"{int(f)}",   "đơn",      "#4CAF80"),
        (c3,"M","Monetary",    f"{m:,.0f}",   "đ",        "#F5C842"),
        (c4,"T","Avg interval",f"{t:.1f}",    "ngày/đơn", "#E5854A"),
    ]
    for col, letter, name, val, unit, accent in cards:
        with col:
            st.markdown(f"""
            <div class="rfmt-card" style="--accent:{accent}">
                <div class="rfmt-letter">{letter}</div>
                <div class="rfmt-name">{name}</div>
                <div class="rfmt-value">{val}<span class="rfmt-unit">{unit}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment + gauge
    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="sec-title">Segment</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="seg-banner" style="--seg-color:{meta['color']}">
            <div class="seg-icon">{meta['icon']}</div>
            <div>
                <div class="seg-lbl">Cluster #{res['cluster']} · {int(model['optimal_k'])} clusters</div>
                <div class="seg-name">{seg}</div>
                <div class="seg-meta">Customer {res['cust_id']} · {res['country']}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-title">Khả năng quay lại</div>', unsafe_allow_html=True)
        if will_return is not None:
            pct   = will_return * 100
            color = "#4CAF80" if pct >= 60 else "#E5854A" if pct >= 35 else "#E24B4A"
            fig   = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":32,"color":"#E8E6F0","family":"DM Mono"}},
                gauge={
                    "axis":     {"range":[0,100],"tickfont":{"color":"#6B6880","size":10}},
                    "bar":      {"color":color,"thickness":0.22},
                    "bgcolor":  "#1A1925","borderwidth":0,
                    "steps":    [{"range":[0,35],"color":"#1F1E28"},
                                 {"range":[35,60],"color":"#201E2A"},
                                 {"range":[60,100],"color":"#1E2026"}],
                    "threshold":{"line":{"color":color,"width":2},"thickness":0.8,"value":pct},
                },
            ))
            fig.update_layout(height=180, margin=dict(t=20,b=10,l=20,r=20),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Cluster này không đủ dữ liệu để dự đoán will_return.")

    # Proba bars
    if all_probas:
        st.markdown('<div class="sec-title">Will_Return theo từng cluster</div>', unsafe_allow_html=True)
        sorted_p = sorted(all_probas.items(), key=lambda x: x[1], reverse=True)
        bars_html = ""
        for i, (lbl, prob) in enumerate(sorted_p):
            pct  = prob * 100
            clr  = BAR_COLORS[i % len(BAR_COLORS)]
            star = "★ " if lbl == seg else ""
            bars_html += f"""
            <div class="proba-row">
                <div class="proba-label">{star}{lbl}</div>
                <div class="proba-bg"><div class="proba-fill" style="width:{pct:.1f}%;background:{clr};"></div></div>
                <div class="proba-pct">{pct:.1f}%</div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="sec-title">Gợi ý hành động</div>
    <div class="rec-box"><strong>{meta['icon']} {seg}</strong> — {meta['action']}</div>
    """, unsafe_allow_html=True)

    # Debug
    with st.expander("🔧 Chi tiết kỹ thuật"):
        st.code(f"""Cluster ID      : {res['cluster']}
Segment         : {seg}
R={r:.0f}  F={f:.0f}  M={m:.1f}  T={t:.1f}
Analysis date   : {ANALYSIS_DT.strftime('%Y-%m-%d')}
Will_Return     : {f'{will_return:.3f}' if will_return else 'N/A'}
Classifier      : XGBClassifier (binary:logistic)
Features (clf)  : Recency, Frequency, Monetary_Past""", language="text")

# ── Empty state ───────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;'>
        <div style='font-size:52px;margin-bottom:16px;'>📊</div>
        <div style='font-size:16px;font-weight:500;color:#6B6880;'>
            Nhập giao dịch bên trái và nhấn <strong style="color:#7C6AF5;">🔍 Phân loại</strong>
        </div>
        <div style='font-size:12px;color:#3D3A50;margin-top:8px;'>
            App tự tính R, F, M, T → KMeans cluster → XGBoost will_return
        </div>
    </div>
    """, unsafe_allow_html=True)
