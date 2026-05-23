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
.rfmt-name { font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: #6B6880; margin-top: 2px; margin-bottom: 1
