import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0d0d14; color: #e8e6f0; }
[data-testid="stSidebar"] { background: #12121e; border-right: 1px solid #2a2a3e; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.segment-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid; border-radius: 12px; padding: 24px;
    margin: 12px 0; transition: transform 0.2s;
}
.segment-card:hover { transform: translateY(-2px); }
.metric-box { background: #1a1a2e; border: 1px solid #2a2a3e; border-radius: 8px; padding: 16px; text-align: center; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.8rem; color: #8884a8; text-transform: uppercase; letter-spacing: 0.1em; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem;
         font-weight: 600; font-family: 'Space Mono', monospace; letter-spacing: 0.05em; }
.stButton>button {
    width: 100%; background: linear-gradient(135deg, #7b5ea7, #4a9eff);
    color: white; border: none; border-radius: 8px; padding: 12px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.05em; font-size: 0.9rem; transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
.upload-hint {
    background: #1a1a2e; border: 2px dashed #3a3a6e; border-radius: 10px;
    padding: 20px; text-align: center; margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Segment config ────────────────────────────────────────────────────────────
SEGMENT_CONFIG = {
    "Champions / VIPs":    {"color": "#ffd700", "bg": "#2a2000", "border": "#ffd700", "icon": "👑", "desc": "Mua thường xuyên, chi tiêu cao, giao dịch gần đây"},
    "Potential Loyalists": {"color": "#4ade80", "bg": "#002a12", "border": "#4ade80", "icon": "⭐", "desc": "Khách hàng tiềm năng, tần suất và giá trị tốt"},
    "Occasional Buyers":   {"color": "#60a5fa", "bg": "#001a2a", "border": "#60a5fa", "icon": "🛒", "desc": "Mua không thường xuyên, cần kích thích thêm"},
    "At-Risk / Dormant":   {"color": "#fb923c", "bg": "#2a1000", "border": "#fb923c", "icon": "⚠️", "desc": "Từng mua tích cực, nhưng đã lâu không quay lại"},
    "Inactive / Churned":  {"color": "#f87171", "bg": "#2a0000", "border": "#f87171", "icon": "💤", "desc": "Không hoạt động trong thời gian dài"},
}

RECOMMENDATIONS = {
    "Champions / VIPs":    ["🎁 Chương trình loyalty VIP", "💌 Ưu tiên sản phẩm mới ra mắt", "🤝 Chăm sóc khách hàng cá nhân hoá"],
    "Potential Loyalists": ["📧 Email nurture sequence", "🏷️ Ưu đãi mua lần 2 & 3", "⭐ Chương trình tích điểm"],
    "Occasional Buyers":   ["🔔 Nhắc nhở mua hàng theo mùa", "💰 Flash sale & giảm giá giới hạn", "📦 Cross-sell & upsell thông minh"],
    "At-Risk / Dormant":   ["💌 Win-back campaign", "🎫 Voucher đặc biệt kích hoạt lại", "📞 Khảo sát lý do không quay lại"],
    "Inactive / Churned":  ["📣 Re-engagement email cuối", "🎯 Retargeting ads", "🔄 Xem xét xoá khỏi danh sách tốn kém"],
}

# ─── Load model ────────────────────────────────────────────────────────────────
def try_load_local():
    """Try loading from disk (local dev only)."""
    for path in [
        "hybrid_customer_segmentation_prediction_model__1_.pkl",
        "/mnt/user-data/uploads/hybrid_customer_segmentation_prediction_model__1_.pkl",
    ]:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

if "model" not in st.session_state:
    st.session_state.model = try_load_local()

model = st.session_state.model
model_loaded = model is not None

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <h1 style="color:#e8e6f0; margin:0; font-size:2rem;">🎯 Customer Segmentation</h1>
  <p style="color:#8884a8; margin:4px 0 0 0; font-family:'DM Sans'; font-size:0.95rem;">
    Hybrid KMeans + XGBoost · RFM Analysis · 5 Phân khúc khách hàng
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model")

    # ── Upload model file ──
    if not model_loaded:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:2rem;">📦</div>
            <div style="font-size:0.85rem;color:#8884a8;margin-top:6px;">
                Upload file <b>.pkl</b> để bắt đầu
            </div>
        </div>
        """, unsafe_allow_html=True)

    uploaded_model = st.file_uploader(
        "Upload file model (.pkl)",
        type=["pkl"],
        help="File: hybrid_customer_segmentation_prediction_model.pkl",
        key="model_uploader",
    )

    if uploaded_model is not None:
        try:
            loaded = pickle.loads(uploaded_model.read())
            st.session_state.model = loaded
            model = loaded
            model_loaded = True
            st.success("✅ Model đã được tải!")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi tải model: {e}")

    # ── Model info ──
    if model_loaded:
        st.markdown(f"""
        <div class="metric-box" style="margin:10px 0;">
            <div class="metric-label">Số phân khúc</div>
            <div class="metric-value" style="color:#7b5ea7;">{model['optimal_k']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Phân khúc:**")
        for seg, cfg in SEGMENT_CONFIG.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #2a2a3e;">
                <span>{cfg['icon']}</span>
                <span style="font-size:0.82rem;color:{cfg['color']};">{seg}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#8884a8; line-height:1.7;">
    <b>RFM Metrics:</b><br>
    • <b>Recency</b>: Ngày kể từ lần mua gần nhất<br>
    • <b>Frequency</b>: Số lần giao dịch<br>
    • <b>Monetary</b>: Tổng chi tiêu<br>
    • <b>T</b>: Tuổi khách hàng (ngày)
    </div>
    """, unsafe_allow_html=True)

# ─── Prediction logic ─────────────────────────────────────────────────────────
def predict_segment(recency, frequency, monetary, T, monetary_past=None):
    scaler = model["scaler"]
    kmeans = model["kmeans_model"]
    classifiers = model["segment_classifiers"]
    cluster_labels = model["cluster_labels"]
    if monetary_past is None:
        monetary_past = monetary
    x_scaled = scaler.transform([[recency, frequency, monetary, T]])
    cluster_id = int(kmeans.predict(x_scaled)[0])
    segment_name = cluster_labels[cluster_id]
    confidence = None
    if cluster_id in classifiers:
        proba = classifiers[cluster_id].predict_proba([[recency, frequency, monetary_past]])[0]
        confidence = float(max(proba))
    return cluster_id, segment_name, confidence

# ─── Show "upload model first" banner if not loaded ───────────────────────────
if not model_loaded:
    st.markdown("""
    <div style="background:#1a0a00;border:1px solid #fb923c;border-radius:12px;
                padding:32px;text-align:center;margin:32px 0;">
        <div style="font-size:3rem;margin-bottom:12px;">📦</div>
        <h3 style="color:#fb923c;font-family:'Space Mono',monospace;margin:0 0 8px 0;">
            Model chưa được tải
        </h3>
        <p style="color:#c8c6d8;margin:0;font-size:0.95rem;">
            Vui lòng upload file <code>hybrid_customer_segmentation_prediction_model.pkl</code>
            trong <b>sidebar bên trái</b> để sử dụng app.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Main tabs (only shown when model loaded) ──────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán đơn lẻ", "📂 Dự đoán hàng loạt (CSV)", "📊 Phân tích phân khúc"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.3], gap="large")

    with col_form:
        st.markdown("#### Nhập thông tin khách hàng")
        recency   = st.number_input("Recency (ngày)", min_value=0, max_value=3000, value=30, step=1,
                                     help="Số ngày kể từ lần mua gần nhất")
        frequency = st.number_input("Frequency (lần)", min_value=1, max_value=500, value=5, step=1,
                                     help="Tổng số lần giao dịch")
        monetary  = st.number_input("Monetary (đơn vị tiền)", min_value=0.0, value=500.0, step=10.0,
                                     help="Tổng giá trị chi tiêu")
        T         = st.number_input("T — Tuổi khách hàng (ngày)", min_value=1, max_value=5000, value=365, step=1,
                                     help="Số ngày từ lần mua đầu tiên đến ngày phân tích")
        monetary_past = st.number_input("Monetary Past (tuỳ chọn)", min_value=0.0, value=monetary, step=10.0,
                                         help="Chi tiêu trong quá khứ (nếu khác Monetary hiện tại)")
        predict_btn = st.button("⚡ Dự đoán phân khúc")

    with col_result:
        if predict_btn:
            try:
                cluster_id, segment_name, confidence = predict_segment(
                    recency, frequency, monetary, T, monetary_past
                )
                cfg = SEGMENT_CONFIG.get(segment_name, {"color":"#aaa","bg":"#111","border":"#aaa","icon":"?","desc":""})

                st.markdown(f"""
                <div class="segment-card" style="border-color:{cfg['border']};background:{cfg['bg']};">
                    <div style="font-size:3rem;margin-bottom:8px;">{cfg['icon']}</div>
                    <div style="font-size:1.5rem;font-weight:700;color:{cfg['color']};
                                font-family:'Space Mono',monospace;margin-bottom:6px;">
                        {segment_name}
                    </div>
                    <div style="color:#c8c6d8;font-size:0.9rem;margin-bottom:16px;">{cfg['desc']}</div>
                    <span class="badge" style="background:{cfg['color']}22;color:{cfg['color']};border:1px solid {cfg['color']};">
                        Cluster #{cluster_id}
                    </span>
                    {f'<span class="badge" style="background:#ffffff11;color:#e8e6f0;border:1px solid #3a3a5e;margin-left:8px;">Confidence: {confidence:.1%}</span>' if confidence else ''}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 💡 Gợi ý hành động")
                for r in RECOMMENDATIONS.get(segment_name, []):
                    st.markdown(f"""
                    <div style="padding:10px 14px;background:#1a1a2e;border-left:3px solid {cfg['color']};
                                border-radius:0 8px 8px 0;margin:6px 0;color:#e8e6f0;font-size:0.9rem;">
                        {r}
                    </div>
                    """, unsafe_allow_html=True)

                # Radar chart
                norm_r = max(0, 1 - recency / 365)
                norm_f = min(frequency / 20, 1)
                norm_m = min(monetary / 5000, 1)
                norm_t = min(T / 1000, 1)
                categories = ["Recency↓", "Frequency", "Monetary", "Customer Age"]
                values = [norm_r, norm_f, norm_m, norm_t]
                fig = go.Figure(go.Scatterpolar(
                    r=values + [values[0]], theta=categories + [categories[0]],
                    fill='toself', line_color=cfg['color'], fillcolor=cfg['color'] + "33",
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="#1a1a2e",
                        radialaxis=dict(visible=True, range=[0,1], gridcolor="#2a2a3e", tickfont=dict(color="#8884a8")),
                        angularaxis=dict(gridcolor="#2a2a3e", tickfont=dict(color="#c8c6d8")),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20, b=20), height=260,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Lỗi dự đoán: {e}")
        else:
            st.markdown("""
            <div style="height:300px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;color:#8884a8;">
                <div style="font-size:3rem;margin-bottom:12px;">🔮</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.9rem;">
                    Nhập thông tin và nhấn Dự đoán
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Tải lên file CSV để dự đoán hàng loạt")
    st.markdown("""
    <div style="background:#1a1a2e;border:1px solid #2a2a3e;border-radius:8px;padding:14px;
                font-size:0.85rem;color:#8884a8;margin-bottom:16px;">
    📋 <b>Cột bắt buộc:</b> <code>Recency, Frequency, Monetary, T</code>
    &nbsp;&nbsp;(tuỳ chọn: <code>Monetary_Past</code>)
    </div>
    """, unsafe_allow_html=True)

    sample_df = pd.DataFrame({
        "Recency":       [10, 45, 200, 350, 90],
        "Frequency":     [25,  8,   2,   1,  5],
        "Monetary":      [5000, 800, 150, 50, 400],
        "T":             [500, 400, 300, 200, 350],
        "Monetary_Past": [5000, 800, 150, 50, 400],
    })
    st.download_button("⬇️ Tải file CSV mẫu", sample_df.to_csv(index=False).encode(),
                       "sample_customers.csv", "text/csv")

    uploaded_csv = st.file_uploader("Chọn file CSV", type=["csv"], key="csv_uploader")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.markdown(f"**{len(df)} dòng** — preview:")
        st.dataframe(df.head(5), use_container_width=True)

        required = {"Recency", "Frequency", "Monetary", "T"}
        if not required.issubset(df.columns):
            st.error(f"Thiếu cột: {required - set(df.columns)}")
        else:
            if st.button("🚀 Chạy dự đoán hàng loạt"):
                with st.spinner("Đang phân tích..."):
                    results = []
                    for _, row in df.iterrows():
                        mp = row.get("Monetary_Past", row["Monetary"])
                        try:
                            cid, seg, conf = predict_segment(
                                row["Recency"], row["Frequency"], row["Monetary"], row["T"], mp
                            )
                            results.append({"Segment": seg, "Cluster_ID": cid,
                                            "Confidence": f"{conf:.1%}" if conf else "—"})
                        except Exception as e:
                            results.append({"Segment": "Error", "Cluster_ID": -1, "Confidence": str(e)})

                result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                st.success(f"✅ Hoàn tất {len(result_df)} khách hàng")

                dist = result_df["Segment"].value_counts().reset_index()
                dist.columns = ["Segment", "Count"]
                colors = [SEGMENT_CONFIG.get(s, {}).get("color", "#aaa") for s in dist["Segment"]]
                fig2 = go.Figure(go.Bar(
                    x=dist["Segment"], y=dist["Count"],
                    marker_color=colors, text=dist["Count"], textposition="outside",
                ))
                fig2.update_layout(
                    title="Phân phối phân khúc",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12121e",
                    font=dict(color="#c8c6d8"),
                    xaxis=dict(gridcolor="#2a2a3e", tickangle=-20),
                    yaxis=dict(gridcolor="#2a2a3e"),
                    margin=dict(t=40, b=40), height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(result_df, use_container_width=True)
                st.download_button("⬇️ Tải kết quả CSV",
                                   result_df.to_csv(index=False).encode(),
                                   "segmented_customers.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Segment analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📊 Tổng quan các phân khúc")

    scaler = model["scaler"]
    km = model["kmeans_model"]
    cluster_labels = model["cluster_labels"]
    feat = model["features_for_scaling"]

    centers_scaled = km.cluster_centers_
    try:
        centers_orig = scaler.inverse_transform(centers_scaled)
    except Exception:
        centers_orig = centers_scaled

    center_df = pd.DataFrame(centers_orig, columns=feat)
    center_df.index = [f"Cluster {i} — {cluster_labels[i]}" for i in range(len(center_df))]

    col_a, col_b = st.columns([1.2, 1], gap="large")
    with col_a:
        st.markdown("**Tâm cluster (giá trị gốc)**")
        st.dataframe(center_df.round(1), use_container_width=True)

    with col_b:
        norm_c = (centers_scaled - centers_scaled.min(0)) / (centers_scaled.max(0) - centers_scaled.min(0) + 1e-9)
        labels_list = [cluster_labels[i] for i in range(len(norm_c))]
        fig3 = go.Figure(go.Heatmap(
            z=norm_c, x=feat, y=labels_list, colorscale="Viridis", showscale=True,
        ))
        fig3.update_layout(
            title="Heatmap Centroids (chuẩn hoá)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8c6d8"), height=280, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("**Đặc điểm & Chiến lược từng phân khúc**")
    cols = st.columns(2)
    for i, (seg, cfg) in enumerate(SEGMENT_CONFIG.items()):
        with cols[i % 2]:
            rec_html = "".join([f"<li style='margin:4px 0;font-size:0.82rem;'>{r}</li>" for r in RECOMMENDATIONS[seg]])
            st.markdown(f"""
            <div class="segment-card" style="border-color:{cfg['border']};background:{cfg['bg']};">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:1.6rem;">{cfg['icon']}</span>
                    <span style="font-weight:700;color:{cfg['color']};font-family:'Space Mono',monospace;font-size:0.95rem;">{seg}</span>
                </div>
                <p style="color:#c8c6d8;font-size:0.85rem;margin:0 0 10px 0;">{cfg['desc']}</p>
                <ul style="margin:0;padding-left:18px;color:#8884a8;">{rec_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    fig4 = go.Figure()
    for i in range(len(centers_orig)):
        seg_name = cluster_labels[i]
        cfg = SEGMENT_CONFIG.get(seg_name, {"color": "#aaa", "icon": "●"})
        r, f, m = centers_orig[i][0], centers_orig[i][1], centers_orig[i][2]
        fig4.add_trace(go.Scatter(
            x=[r], y=[f], mode="markers+text",
            marker=dict(size=max(10, min(abs(m)/100, 50)), color=cfg["color"], opacity=0.85,
                        line=dict(width=2, color="#ffffff33")),
            text=[f"{cfg['icon']} {seg_name}"], textposition="top center", name=seg_name,
        ))
    fig4.update_layout(
        title="Cluster Centres — Recency vs Frequency (bubble = Monetary)",
        xaxis_title="Recency (ngày)", yaxis_title="Frequency (lần mua)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12121e",
        font=dict(color="#c8c6d8"),
        xaxis=dict(gridcolor="#2a2a3e"), yaxis=dict(gridcolor="#2a2a3e"),
        height=380, showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
<hr style="border-color:#2a2a3e;margin-top:32px;">
<div style="text-align:center;color:#8884a8;font-size:0.78rem;padding:8px 0 16px;">
    Hybrid Customer Segmentation · KMeans + XGBoost · RFM Model
</div>
""", unsafe_allow_html=True)
