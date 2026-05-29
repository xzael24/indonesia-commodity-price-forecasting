# -*- coding: utf-8 -*-
# app.py - Streamlit Dashboard Prediksi Harga Kebutuhan Pokok
"""
Jalankan: streamlit run app.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, PRED_LEN, SEQ_LEN
from src.model import get_model

# Page config
st.set_page_config(
    page_title="Prediksi Harga Kebutuhan Pokok",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/line-chart.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SVG Icons
SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>"""

SVG_BASKET = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>"""

SVG_BRAIN = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/></svg>"""

SVG_CALENDAR = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>"""

SVG_INFO = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>"""

SVG_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""

SVG_ALERT_TRIANGLE = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""

SVG_ALERT_SHIELD = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

SVG_TROPHY = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.45 1-1 1H4v2h16v-2h-5c-.55 0-1-.45-1-1v-2.34"/><path d="M12 2a6 6 0 0 1 6 6v3.5c0 1.63-.84 3.06-2.12 3.9A6.05 6.05 0 0 1 12 16a6.05 6.05 0 0 1-3.88-.6c-1.28-.84-2.12-2.27-2.12-3.9V8a6 6 0 0 1 6-6z"/></svg>"""

SVG_MAIN_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>"""

SVG_UP = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>"""

SVG_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>"""

SVG_FLAT = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>"""

# ── Global CSS Injection ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Typography overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F17 !important;
        border-right: 1px solid #1F2937 !important;
        padding-top: 1.5rem;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #F9FAFB !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    
    /* Styled Selectboxes and inputs */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
        border: 1px solid #1F2937 !important;
        background-color: #111827 !important;
        transition: border-color 0.2s ease;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: #3B82F6 !important;
    }

    /* Slider customization */
    .stSlider [data-disabled="false"] {
        color: #3B82F6 !important;
    }
    
    /* Clean Divider */
    hr {
        border-color: #1F2937 !important;
        margin: 2rem 0 !important;
    }

    /* Custom CSS for Dataframe display */
    .stDataFrame {
        border: 1px solid #1F2937 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Tab Custom Styling */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #9CA3AF !important;
        border-bottom: 2px solid transparent !important;
        background-color: transparent !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #F9FAFB !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom-color: #3B82F6 !important;
        font-weight: 600 !important;
    }
    
    /* Premium details */
    .custom-metric-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    
    .custom-metric-card:hover {
        border-color: #3B82F6;
    }
    
    .custom-metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .custom-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F9FAFB;
        line-height: 1.2;
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.01em;
    }
    
    .custom-metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-weight: 500;
    }
    
    .delta-positive {
        color: #F43F5E;
    }
    
    .delta-negative {
        color: #10B981;
    }
    
    .delta-neutral {
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────

@st.cache_resource
def load_scaler():
    path = os.path.join(DATA_PROCESSED_DIR, "scaler.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_cached(model_name: str, commodity_name: str, input_size: int):
    ckpt = os.path.join(MODEL_DIR, f"{model_name}_{commodity_name}_best.pt")
    if not os.path.exists(ckpt):
        # Fallback to general model if commodity-specific model doesn't exist
        ckpt_fallback = os.path.join(MODEL_DIR, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_fallback):
            return None
        ckpt = ckpt_fallback
    model = get_model(model_name, input_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_local_prices():
    path = os.path.join(DATA_RAW_DIR, "pihps_harga.csv")
    if not os.path.exists(path):
        path = os.path.join(DATA_RAW_DIR, "bapanas_harga.csv")  # fallback nama lama
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

@st.cache_data
def load_results():
    path = os.path.join(MODEL_DIR, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def predict_next_days(model, last_sequence: np.ndarray, scaler, target_idx: int, n_days: int = PRED_LEN):
    """Prediksi n_days ke depan dari last_sequence."""
    model.eval()
    x = torch.FloatTensor(last_sequence).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(x).numpy()[0]

    # Inverse transform
    n_feat = scaler.n_features_in_
    dummy  = np.zeros((len(pred_scaled), n_feat))
    dummy[:, target_idx] = pred_scaled
    pred_prices = scaler.inverse_transform(dummy)[:, target_idx]
    return pred_prices[:n_days]


def render_custom_metric(label, value_str, delta, delta_str, svg_icon):
    """Render a clean custom metric card instead of Streamlit's default."""
    if delta > 0:
        delta_class = "delta-positive"
        trend_svg = SVG_UP
    elif delta < 0:
        delta_class = "delta-negative"
        trend_svg = SVG_DOWN
    else:
        delta_class = "delta-neutral"
        trend_svg = SVG_FLAT
        
    html = f"""
    <div class="custom-metric-card">
        <div class="custom-metric-label">
            <span style="display: flex; align-items: center; justify-content: center; color: #9CA3AF;">{svg_icon}</span>
            <span>{label}</span>
        </div>
        <div class="custom-metric-value">{value_str}</div>
        <div class="custom-metric-delta {delta_class}">
            <span style="display: flex; align-items: center; justify-content: center;">{trend_svg}</span>
            <span>{delta_str}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_alert_success(message):
    """Success banner - soft emerald green theme."""
    html = f"""
    <div style="
        background-color: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px;
        padding: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        color: #F9FAFB;
        margin-bottom: 1.5rem;
    ">
        <div style="color: #10B981; flex-shrink: 0; display: flex; align-items: center; margin-top: 0.125rem;">
            {SVG_CHECK}
        </div>
        <div style="font-size: 0.875rem; line-height: 1.5; color: #E5E7EB; font-family: 'Inter', sans-serif;">
            {message}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_alert_warning(message):
    """Warning banner - soft amber theme."""
    html = f"""
    <div style="
        background-color: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        color: #F9FAFB;
        margin-bottom: 1.5rem;
    ">
        <div style="color: #F59E0B; flex-shrink: 0; display: flex; align-items: center; margin-top: 0.125rem;">
            {SVG_ALERT_TRIANGLE}
        </div>
        <div style="font-size: 0.875rem; line-height: 1.5; color: #E5E7EB; font-family: 'Inter', sans-serif;">
            {message}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_alert_danger(message):
    """Danger banner - soft rose red theme."""
    html = f"""
    <div style="
        background-color: rgba(244, 63, 94, 0.05);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 8px;
        padding: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        color: #F9FAFB;
        margin-bottom: 1.5rem;
    ">
        <div style="color: #F43F5E; flex-shrink: 0; display: flex; align-items: center; margin-top: 0.125rem;">
            {SVG_ALERT_SHIELD}
        </div>
        <div style="font-size: 0.875rem; line-height: 1.5; color: #E5E7EB; font-family: 'Inter', sans-serif;">
            {message}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def style_plotly_chart(fig):
    """Style Plotly charts to look like native high-precision financial platforms."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=11, color="#9CA3AF"),
        margin=dict(l=40, r=15, t=15, b=30),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#1F2937",
            font=dict(family="Inter, sans-serif", size=12, color="#F9FAFB")
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#9CA3AF")
        )
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#1F2937",
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#9CA3AF")
    )
    fig.update_yaxes(
        showgrid=True,
        linecolor="#1F2937",
        gridcolor="rgba(255, 255, 255, 0.05)",
        tickfont=dict(color="#9CA3AF"),
        zeroline=False
    )


def load_external_data():
    path_ext = os.path.join(DATA_RAW_DIR, "external_factors.csv")
    path_weather = os.path.join(DATA_RAW_DIR, "bmkg_cuaca.csv")
    if not os.path.exists(path_ext) or not os.path.exists(path_weather):
        return None, None
    df_ext = pd.read_csv(path_ext, index_col=0, parse_dates=True)
    df_wea = pd.read_csv(path_weather, index_col=0, parse_dates=True)
    return df_ext, df_wea


def simulate_scenario(last_sequence: np.ndarray, scaler, cols: list, usd_change_pct: float, oil_change_pct: float, rainfall_change_mm: float) -> np.ndarray:
    """Adjust features in last_sequence based on scenario shifts."""
    seq_copy = last_sequence.copy()
    raw_seq = scaler.inverse_transform(seq_copy)
    df_temp = pd.DataFrame(raw_seq, columns=cols)
    
    if usd_change_pct != 0.0:
        if "USD_IDR" in df_temp.columns:
            df_temp["USD_IDR"] *= (1 + usd_change_pct / 100.0)
            
    if oil_change_pct != 0.0:
        if "Minyak" in df_temp.columns:
            df_temp["Minyak"] *= (1 + oil_change_pct / 100.0)
            
    if rainfall_change_mm != 0.0:
        if "curah_hujan_mm" in df_temp.columns:
            df_temp["curah_hujan_mm"] = np.clip(df_temp["curah_hujan_mm"] + rainfall_change_mm, 0, None)
            
    adjusted_scaled = scaler.transform(df_temp.values)
    return adjusted_scaled


def check_ramadan_proximity(pred_dates):
    """Cek apakah ada tanggal prediksi yang jatuh di periode Ramadan."""
    ramadan_windows = [
        ("2020-04-23", "2020-05-23"), ("2021-04-12", "2021-05-12"),
        ("2022-04-02", "2022-05-01"), ("2023-03-22", "2023-04-20"),
        ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
        ("2026-02-18", "2026-03-19"),
    ]
    for start_str, end_str in ramadan_windows:
        start = datetime.strptime(start_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_str, "%Y-%m-%d").date()
        
        for d in pred_dates:
            d_date = d.date() if isinstance(d, datetime) else d
            if start <= d_date <= end:
                return True, start, end
    return False, None, None


# ── Sidebar ────────────────────────────────────────────────────────────────

# Clean container layout for the themed application logo
st.sidebar.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 1.5rem; margin-top: 0.5rem; padding: 1rem; border: 1px solid #1F2937; border-radius: 12px; background-color: #111827; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="70" height="70" style="margin-bottom: 0.5rem;">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#10B981;stop-opacity:1" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <circle cx="50" cy="50" r="44" fill="none" stroke="url(#grad1)" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.3"/>
        <path d="M 22 62 L 40 48 L 56 56 L 78 30" fill="none" stroke="url(#grad1)" stroke-width="2.5" stroke-linecap="round" filter="url(#glow)"/>
        <circle cx="78" cy="30" r="3.5" fill="#10B981" />
        <circle cx="40" cy="48" r="2" fill="#3B82F6" />
        <path d="M32 42 h36 l-4 22 h-28 z" fill="none" stroke="#F9FAFB" stroke-width="2" stroke-linejoin="round"/>
        <path d="M42 42 V 35 A 8 8 0 0 1 58 35 V 42" fill="none" stroke="#F9FAFB" stroke-width="1.75" stroke-linecap="round"/>
        <path d="M50 48 C 54 44, 56 48, 56 52 C 56 56, 52 56, 50 56 C 48 56, 44 56, 44 52 C 44 48, 46 44, 50 48 Z" fill="url(#grad1)" opacity="0.8"/>
        <path d="M50 56 V 48" fill="none" stroke="#F9FAFB" stroke-width="1.25" stroke-linecap="round"/>
    </svg>
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.95rem; font-weight: 700; background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; letter-spacing: 0.05em; text-transform: uppercase;">
        AGRI-FORECAST
    </div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #6B7280; text-align: center; margin-top: 0.15rem; font-weight: 500;">
        Deep Learning AI Engine
    </div>
</div>
""", unsafe_allow_html=True)

# Settings header
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.25rem;">
    <span style="color: #9CA3AF; display: flex; align-items: center;">{SVG_SETTINGS}</span>
    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 600; color: #F9FAFB; font-family: 'Plus Jakarta Sans', sans-serif;">Pengaturan</h3>
</div>
""", unsafe_allow_html=True)

commodity_options = ["Beras", "Minyak Goreng", "Telur Ayam", "Cabai Merah", "Daging Ayam"]

# Commodity selector
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 0.75rem;">
    <span style="color: #9CA3AF; display: flex; align-items: center;">{SVG_BASKET}</span>
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Pilih Komoditas</span>
</div>
""", unsafe_allow_html=True)
selected_commodity = st.sidebar.selectbox("Pilih Komoditas", commodity_options, label_visibility="collapsed")

model_options = ["tft", "lstm", "gru"]

# Model selector
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 1rem;">
    <span style="color: #9CA3AF; display: flex; align-items: center;">{SVG_BRAIN}</span>
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Model Prediksi</span>
</div>
""", unsafe_allow_html=True)
selected_model = st.sidebar.selectbox(
    "Model",
    model_options,
    format_func=lambda x: {"tft": "TFT (Terbaik)", "lstm": "LSTM", "gru": "GRU"}[x],
    label_visibility="collapsed"
)

# Horizon slider
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 1rem;">
    <span style="color: #9CA3AF; display: flex; align-items: center;">{SVG_CALENDAR}</span>
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Horizon Prediksi (hari)</span>
</div>
""", unsafe_allow_html=True)
pred_horizon = st.sidebar.slider("Horizon Prediksi", 1, PRED_LEN, PRED_LEN, label_visibility="collapsed")

st.sidebar.markdown("<hr style='margin: 1rem 0; border-color: #1F2937;' />", unsafe_allow_html=True)

# Inisialisasi session state untuk slider simulasi
if "sim_usd_val" not in st.session_state:
    st.session_state.sim_usd_val = 0.0
if "sim_oil_val" not in st.session_state:
    st.session_state.sim_oil_val = 0.0
if "sim_rain_val" not in st.session_state:
    st.session_state.sim_rain_val = 0.0

# ── Simulasi Skenario ──────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
    <span style="color: #9CA3AF; display: flex; align-items: center;">{SVG_SETTINGS}</span>
    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 600; color: #F9FAFB; font-family: 'Plus Jakarta Sans', sans-serif;">Simulasi Skenario (What-If)</h3>
</div>
""", unsafe_allow_html=True)

# Preset Skenario Selector
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem;">
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Preset Skenario Makro</span>
</div>
""", unsafe_allow_html=True)

preset_options = {
    "Kustom (Atur Sendiri)": {"usd": None, "oil": None, "rain": None},
    "El Niño (Kekeringan Ekstrem)": {"usd": 0.0, "oil": 0.0, "rain": -10.0},
    "Krisis Energi Global": {"usd": 8.0, "oil": 40.0, "rain": 0.0},
    "Depresiasi Rupiah Ekstrem": {"usd": 15.0, "oil": 0.0, "rain": 0.0},
    "Normal / Stabilisasi": {"usd": 0.0, "oil": 0.0, "rain": 0.0}
}

selected_preset = st.sidebar.selectbox(
    "Preset Skenario",
    options=list(preset_options.keys()),
    label_visibility="collapsed",
    key="preset_select"
)

# Jika preset berubah dan bukan Kustom, paksa update session state slider
if selected_preset != "Kustom (Atur Sendiri)":
    vals = preset_options[selected_preset]
    st.session_state.sim_usd_val = vals["usd"]
    st.session_state.sim_oil_val = vals["oil"]
    st.session_state.sim_rain_val = vals["rain"]
    disabled_sliders = True
else:
    disabled_sliders = False

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 0.5rem;">
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Kurs USD/IDR (%)</span>
</div>
""", unsafe_allow_html=True)
sim_usd = st.sidebar.slider(
    "USD/IDR", -20.0, 20.0, step=0.5, format="%+.1f%%",
    key="sim_usd_val", disabled=disabled_sliders, label_visibility="collapsed"
)

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 0.75rem;">
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Harga Minyak Dunia (%)</span>
</div>
""", unsafe_allow_html=True)
sim_oil = st.sidebar.slider(
    "Minyak Dunia", -50.0, 50.0, step=1.0, format="%+.1f%%",
    key="sim_oil_val", disabled=disabled_sliders, label_visibility="collapsed"
)

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.35rem; margin-top: 0.75rem;">
    <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF;">Curah Hujan (mm/hari)</span>
</div>
""", unsafe_allow_html=True)
sim_rain = st.sidebar.slider(
    "Curah Hujan", -15.0, 15.0, step=0.5, format="%+0.1f mm",
    key="sim_rain_val", disabled=disabled_sliders, label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='margin: 1.5rem 0; border-color: #1F2937;' />", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.5rem;">
    <span style="color: #3B82F6; display: flex; align-items: center;">{SVG_INFO}</span>
    <span style="font-size: 0.85rem; font-weight: 600; color: #F9FAFB; font-family: 'Plus Jakarta Sans', sans-serif;">Tentang Proyek</span>
</div>
<div style="font-size: 0.8rem; line-height: 1.5; color: #9CA3AF; font-family: 'Inter', sans-serif;">
    Prediksi harga kebutuhan pokok menggunakan Deep Learning berbasis Multivariate Time Series dengan integrasi faktor ekonomi global.
</div>
""", unsafe_allow_html=True)

# ── Main Content ───────────────────────────────────────────────────────────

st.markdown(f"""
<div style="margin-bottom: 2rem; margin-top: 0.5rem;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.35rem;">
        <span style="color: #3B82F6; display: flex; align-items: center;">{SVG_MAIN_LOGO}</span>
        <h1 style="margin: 0; font-size: 2.15rem; font-weight: 700; color: #F9FAFB; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif;">
            Prediksi Harga Kebutuhan Pokok Indonesia
        </h1>
    </div>
    <p style="margin: 0; font-size: 0.9rem; color: #9CA3AF; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 0.5rem; padding-left: 0.25rem;">
        <span>Berbasis Deep Learning</span>
        <span style="color: #374151;">&bull;</span>
        <span>Multivariate Time Series</span>
        <span style="color: #374151;">&bull;</span>
        <span>Integrasi Faktor Ekonomi Global</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediksi & Simulasi", "Faktor Eksternal", "Performa Model", "Tentang Proyek", "Pusat Kontrol Pipeline"])

# ── Tab 1: Prediksi ─────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    df_local = load_local_prices()
    scaler_data = load_scaler()

    if df_local is None or scaler_data is None:
        render_alert_warning(
            "<strong>Data atau model belum tersedia</strong>. Silakan jalankan pipeline terlebih dahulu di terminal menggunakan command di bawah ini."
        )
        st.code("python run_pipeline.py", language="bash")
    else:
        scaler = scaler_data["scaler"]
        cols   = scaler_data["columns"]

        # Tampilkan data historis
        with col1:
            st.markdown("""
            <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 1rem; margin-top: 0.25rem; font-family: 'Plus Jakarta Sans', sans-serif;">
                Harga Historis
            </h3>
            """, unsafe_allow_html=True)
            
            if selected_commodity in df_local.columns:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=df_local.index[-180:],
                    y=df_local[selected_commodity].iloc[-180:],
                    mode="lines", name="Historis",
                    line=dict(color="#3B82F6", width=2)
                ))
                style_plotly_chart(fig_hist)
                fig_hist.update_layout(
                    height=320,
                    margin=dict(l=40, r=10, t=10, b=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown("""
            <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 1rem; margin-top: 0.25rem; font-family: 'Plus Jakarta Sans', sans-serif;">
                Harga Terakhir
            </h3>
            """, unsafe_allow_html=True)
            
            if selected_commodity in df_local.columns:
                latest = df_local[selected_commodity].dropna().iloc[-1]
                prev7  = df_local[selected_commodity].dropna().iloc[-8]
                delta  = latest - prev7
                
                # Format strings
                val_str = f"Rp {latest:,.0f}"
                if delta > 0:
                    delta_str = f"+Rp {delta:,.0f} vs 7 hari lalu"
                elif delta < 0:
                    delta_str = f"-Rp {abs(delta):,.0f} vs 7 hari lalu"
                else:
                    delta_str = "Stabil vs 7 hari lalu"
                
                render_custom_metric(
                    label=selected_commodity,
                    value_str=val_str,
                    delta=delta,
                    delta_str=delta_str,
                    svg_icon=SVG_BASKET
                )

        st.markdown("<hr style='border-color: #1F2937; margin: 1.5rem 0;' />", unsafe_allow_html=True)

        # Prediksi
        st.markdown(f"""
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 1.25rem; font-family: 'Plus Jakarta Sans', sans-serif;">
            Proyeksi Tren {pred_horizon} Hari ke Depan
        </h3>
        """, unsafe_allow_html=True)

        X_test_path = os.path.join(DATA_PROCESSED_DIR, "X_test.npy")
        if os.path.exists(X_test_path):
            X_test = np.load(X_test_path)
            input_size = X_test.shape[2]
            model = load_model_cached(selected_model, selected_commodity, input_size)

            if model is None:
                render_alert_warning(
                    f"Model <strong>{selected_model.upper()}</strong> belum ditraining. Silakan jalankan training pipeline terlebih dahulu."
                )
            else:
                target_idx = cols.index(selected_commodity) if selected_commodity in cols else 0
                last_seq   = X_test[-1]
                pred = predict_next_days(model, last_seq, scaler, target_idx, n_days=pred_horizon)

                # What-If Scenario prediction
                is_scenario_active = (sim_usd != 0.0 or sim_oil != 0.0 or sim_rain != 0.0)
                if is_scenario_active:
                    sim_seq = simulate_scenario(last_seq, scaler, cols, sim_usd, sim_oil, sim_rain)
                    pred_sim = predict_next_days(model, sim_seq, scaler, target_idx, n_days=pred_horizon)
                else:
                    pred_sim = None

                # Tanggal prediksi
                last_date = df_local.index[-1] if hasattr(df_local.index[-1], 'date') else datetime.today()
                pred_dates = [last_date + timedelta(days=i + 1) for i in range(len(pred))]

                # Cek Ramadan Proximity
                is_ram, ram_start, ram_end = check_ramadan_proximity(pred_dates)

                # Gabungkan historis + prediksi
                hist_30 = df_local[selected_commodity].iloc[-30:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=hist_30.index, y=hist_30.values,
                    mode="lines", name="Historis (30 Hari)",
                    line=dict(color="#3B82F6", width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=pred_dates, y=pred,
                    mode="lines+markers", name="Proyeksi Baseline",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                    marker=dict(size=5, color="#EF4444")
                ))

                if pred_sim is not None:
                    fig_pred.add_trace(go.Scatter(
                        x=pred_dates, y=pred_sim,
                        mode="lines+markers", name="Proyeksi Skenario",
                        line=dict(color="#F59E0B", width=2, dash="dot"),
                        marker=dict(size=5, color="#F59E0B")
                    ))

                # Confidence band dinamis berbasis RMSE aktual (95% CI)
                results_data = load_results()
                rmse_val = None
                if results_data and selected_commodity in results_data:
                    commodity_res = results_data[selected_commodity]
                    if selected_model in commodity_res and isinstance(commodity_res[selected_model], dict):
                        rmse_val = commodity_res[selected_model].get("RMSE", None)
                
                if rmse_val is not None:
                    # 95% CI: +/- 1.96 * RMSE
                    y_upper = list(pred + 1.96 * rmse_val)
                    y_lower = list(np.clip(pred - 1.96 * rmse_val, 0, None))
                    ci_name = "Rentang Estimasi (95% CI)"
                else:
                    # Fallback ke ±5% jika data performa tidak ditemukan
                    y_upper = list(pred * 1.05)
                    y_lower = list(pred * 0.95)
                    ci_name = "Rentang Estimasi (±5% Fallback)"

                fig_pred.add_trace(go.Scatter(
                    x=pred_dates + pred_dates[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself", fillcolor="rgba(239,68,68,0.04)",
                    line=dict(color="rgba(0,0,0,0)"), name=ci_name
                ))

                if is_ram:
                    fig_pred.add_vrect(
                        x0=max(ram_start, pred_dates[0].date() if isinstance(pred_dates[0], datetime) else pred_dates[0]),
                        x1=min(ram_end, pred_dates[-1].date() if isinstance(pred_dates[-1], datetime) else pred_dates[-1]),
                        fillcolor="rgba(16, 185, 129, 0.07)",
                        layer="below", line_width=0,
                        annotation_text="Periode Ramadan",
                        annotation_position="top left",
                        annotation_font=dict(size=10, color="#10B981")
                    )

                style_plotly_chart(fig_pred)
                fig_pred.update_layout(
                    height=360,
                    margin=dict(l=40, r=10, t=10, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Split projection details
                col_tbl, col_alrt = st.columns([1, 1])
                
                with col_tbl:
                    tbl_data = {
                        "Tanggal": [d.strftime("%d %b %Y") for d in pred_dates],
                        f"Proyeksi Baseline": [f"Rp {p:,.0f}" for p in pred]
                    }
                    if pred_sim is not None:
                        tbl_data["Proyeksi Skenario"] = [f"Rp {p:,.0f}" for p in pred_sim]
                    df_pred = pd.DataFrame(tbl_data)
                    st.dataframe(df_pred, use_container_width=True, hide_index=True)

                    # Export button
                    df_download = pd.DataFrame({
                        "Tanggal": [d.strftime("%Y-%m-%d") for d in pred_dates],
                        "Komoditas": selected_commodity,
                        "Proyeksi Baseline (Rp)": pred.round(0)
                    })
                    if pred_sim is not None:
                        df_download["Proyeksi Skenario (Rp)"] = pred_sim.round(0)
                    
                    csv_bytes = df_download.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh Laporan Proyeksi (CSV)",
                        data=csv_bytes,
                        file_name=f"proyeksi_{selected_commodity}_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_alrt:
                    if is_ram:
                        render_alert_warning(
                            f"<strong>Deteksi HBKN (Ramadan/Lebaran):</strong> Rentang proyeksi bersinggungan dengan periode Ramadan ({ram_start.strftime('%d %b')} - {ram_end.strftime('%d %b %Y')}). Secara historis, terjadi lonjakan permintaan pada komoditas pangan pokok."
                        )

                    # Early warning evaluation (based on baseline or scenario)
                    eval_price = pred_sim.max() if pred_sim is not None else pred.max()
                    current_price = df_local[selected_commodity].iloc[-1]
                    
                    if eval_price > current_price * 1.10:
                        render_alert_danger(
                            f"<strong>Peringatan Dini</strong>: Terdeteksi potensi lonjakan harga signifikan pada <strong>{selected_commodity}</strong> sebesar lebih dari <strong>10%</strong> dalam {pred_horizon} hari ke depan. Harap waspada terhadap ketidakstabilan pasokan."
                        )
                    elif eval_price > current_price * 1.05:
                        render_alert_warning(
                            f"<strong>Perhatian Proyeksi</strong>: Harga komoditas <strong>{selected_commodity}</strong> diproyeksikan mengalami kenaikan terukur sekitar <strong>5%</strong> dalam {pred_horizon} hari ke depan."
                        )
                    else:
                        render_alert_success(
                            f"Proyeksi komoditas pangan menunjukkan harga <strong>{selected_commodity}</strong> diperkirakan dalam kondisi relatif <strong>stabil dan aman</strong> dalam {pred_horizon} hari ke depan."
                        )
        else:
            render_alert_warning("Jalankan <code>python run_pipeline.py</code> untuk memproses dataset dan melatih model.")


# ── Tab 2: Faktor Eksternal ─────────────────────────────────────────────────
with tab2:
    df_ext, df_wea = load_external_data()
    if df_local is None or df_ext is None or df_wea is None:
        render_alert_warning("Data faktor eksternal belum tersedia. Silakan jalankan training pipeline terlebih dahulu.")
    else:
        st.markdown("""
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 1.25rem; font-family: 'Plus Jakarta Sans', sans-serif;">
            Analisis Faktor Eksternal (Global Market & Cuaca)
        </h3>
        """, unsafe_allow_html=True)
        
        # Let's align indices and compute correlation
        df_local.index = pd.to_datetime(df_local.index)
        df_ext.index = pd.to_datetime(df_ext.index)
        df_wea.index = pd.to_datetime(df_wea.index)
        
        df_all = df_local.join(df_ext, how="inner").join(df_wea, how="inner")
        
        local_cols = ["Beras", "Minyak Goreng", "Telur Ayam", "Cabai Merah", "Daging Ayam"]
        ext_cols = ["Gandum", "Kedelai", "Jagung", "Minyak", "USD_IDR", "curah_hujan_mm", "suhu_c"]
        
        # Filter only existing columns
        local_cols = [c for c in local_cols if c in df_all.columns]
        ext_cols = [c for c in ext_cols if c in df_all.columns]
        
        if local_cols and ext_cols:
            corr = df_all[local_cols + ext_cols].corr()
            corr_sub = corr.loc[local_cols, ext_cols]
            
            col_heat, col_desc = st.columns([3, 2])
            with col_heat:
                st.markdown("##### Matriks Korelasi (Pearson)")
                fig_corr = px.imshow(
                    corr_sub,
                    labels=dict(x="Faktor Eksternal", y="Komoditas Lokal", color="Korelasi"),
                    x=ext_cols,
                    y=local_cols,
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                    text_auto=".2f"
                )
                style_plotly_chart(fig_corr)
                fig_corr.update_layout(height=300, margin=dict(l=40, r=10, t=10, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with col_desc:
                st.markdown("##### Interpretasi Korelasi")
                st.markdown("""
                * **Nilai Mendekati +1:** Korelasi positif kuat (misal: jika kurs USD/IDR naik, harga komoditas lokal cenderung naik).
                * **Nilai Mendekati -1:** Korelasi negatif kuat (misal: jika curah hujan tinggi, suhu cenderung turun, atau komoditas tertentu mengalami penurunan harga karena masa panen melimpah).
                * **Nilai Mendekati 0:** Hubungan linier sangat lemah atau tidak ada korelasi langsung.
                """)
                
        if local_cols and ext_cols and selected_commodity in corr_sub.index:
            st.markdown("<hr style='border-color: #1F2937; margin: 1.5rem 0;' />", unsafe_allow_html=True)
            st.markdown(f"""
            <h4 style="font-size: 1.05rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
                Explainable AI (XAI) & Dampak Fitur terhadap {selected_commodity}
            </h4>
            <p style="font-size: 0.85rem; color: #9CA3AF; font-family: 'Inter', sans-serif; margin-bottom: 1.25rem;">
                Visualisasi di bawah mendeteksi kekuatan dan arah hubungan (koefisien korelasi) faktor eksternal makroekonomi dan cuaca secara khusus terhadap pergerakan harga <strong>{selected_commodity}</strong>. Faktor bernilai positif mendorong inflasi harga, sementara faktor bernilai negatif meredam harga.
            </p>
            """, unsafe_allow_html=True)

            col_xai_plot, col_xai_desc = st.columns([3, 2])
            
            with col_xai_plot:
                commodity_corr = corr_sub.loc[selected_commodity].reset_index()
                commodity_corr.columns = ["Faktor Eksternal", "Koefisien Korelasi"]
                
                # Sort values for ascending layout
                commodity_corr = commodity_corr.sort_values(by="Koefisien Korelasi", ascending=True)
                
                commodity_corr["Arah Hubungan"] = commodity_corr["Koefisien Korelasi"].apply(
                    lambda x: "Mendorong Harga Naik (+)" if x > 0 else "Menekan Harga Turun (-)"
                )
                
                fig_impact = px.bar(
                    commodity_corr,
                    x="Koefisien Korelasi",
                    y="Faktor Eksternal",
                    color="Arah Hubungan",
                    orientation="h",
                    color_discrete_map={
                        "Mendorong Harga Naik (+)": "#F43F5E",  # Soft rose red
                        "Menekan Harga Turun (-)": "#10B981"   # Soft emerald green
                    }
                )
                style_plotly_chart(fig_impact)
                fig_impact.update_layout(
                    height=260,
                    margin=dict(l=40, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_impact, use_container_width=True)

            with col_xai_desc:
                st.markdown(f"""
                <h5 style="font-size: 0.95rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
                    <span style="color: #3B82F6; display: inline-flex; align-items: center;">{SVG_INFO}</span>
                    <span>Temuan Analitis</span>
                </h5>
                """, unsafe_allow_html=True)
                
                pos_factors = commodity_corr[commodity_corr["Koefisien Korelasi"] > 0]
                neg_factors = commodity_corr[commodity_corr["Koefisien Korelasi"] < 0]
                
                strongest_pos = pos_factors.iloc[-1] if not pos_factors.empty else None
                strongest_neg = neg_factors.iloc[0] if not neg_factors.empty else None
                
                narrative_html = f"<div style='font-size: 0.85rem; line-height: 1.6; color: #9CA3AF; font-family: \"Inter\", sans-serif;'>"
                narrative_html += f"<p style='margin-top:0;'>Hasil analisis korelasi data historis menunjukkan bahwa harga <strong>{selected_commodity}</strong> memiliki pola hubungan sebagai berikut:</p>"
                narrative_html += "<ul style='list-style: none; padding-left: 0; display: flex; flex-direction: column; gap: 0.75rem; margin-top: 1rem; margin-bottom: 0;'>"
                
                if strongest_pos is not None and strongest_pos["Koefisien Korelasi"] > 0.15:
                    narrative_html += f"""
                    <li style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="color: #F43F5E; flex-shrink: 0; display: inline-flex; align-items: center; margin-top: 0.2rem;">{SVG_UP}</span>
                        <span><strong>{strongest_pos['Faktor Eksternal']}</strong> memiliki korelasi positif paling dominan (<strong>r = {strongest_pos['Koefisien Korelasi']:.2f}</strong>). Kenaikan pada variabel ini cenderung mentransmisikan tekanan inflasi yang meningkatkan harga {selected_commodity}.</span>
                    </li>
                    """
                
                if strongest_neg is not None and strongest_neg["Koefisien Korelasi"] < -0.15:
                    narrative_html += f"""
                    <li style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="color: #10B981; flex-shrink: 0; display: inline-flex; align-items: center; margin-top: 0.2rem;">{SVG_DOWN}</span>
                        <span><strong>{strongest_neg['Faktor Eksternal']}</strong> bertindak sebagai penyeimbang dengan korelasi negatif terkuat (<strong>r = {strongest_neg['Koefisien Korelasi']:.2f}</strong>). Hal ini mengindikasikan bahwa ketika variabel tersebut meningkat, harga {selected_commodity} cenderung bergerak turun.</span>
                    </li>
                    """
                
                if (strongest_pos is None or strongest_pos["Koefisien Korelasi"] <= 0.15) and (strongest_neg is None or strongest_neg["Koefisien Korelasi"] >= -0.15):
                    narrative_html += f"""
                    <li style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="color: #9CA3AF; flex-shrink: 0; display: inline-flex; align-items: center; margin-top: 0.2rem;">{SVG_FLAT}</span>
                        <span><strong>Pengaruh Global & Cuaca Moderat:</strong> Hubungan langsung dengan variabel makroekonomi global & cuaca terpantau relatif lemah (di bawah 0.15). Harga {selected_commodity} di pasar eceran tampaknya lebih dominan dipengaruhi oleh faktor penawaran-permintaan domestik lokal, rantai distribusi regional, atau intervensi stabilisasi pasokan oleh pemerintah.</span>
                    </li>
                    """
                
                narrative_html += "</ul></div>"
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(59, 130, 246, 0.02);
                    border: 1px solid #1F2937;
                    border-radius: 12px;
                    padding: 1.25rem;
                    margin-top: 0.5rem;
                ">
                    {narrative_html}
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<hr style='border-color: #1F2937; margin: 1.5rem 0;' />", unsafe_allow_html=True)
        
        # Show some of the external factors plots
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### Nilai Tukar Rupiah (USD/IDR)")
            if "USD_IDR" in df_ext.columns:
                fig_usd = px.line(df_ext, y="USD_IDR", color_discrete_sequence=["#10B981"])
                style_plotly_chart(fig_usd)
                fig_usd.update_layout(height=220, margin=dict(l=40, r=10, t=10, b=10))
                st.plotly_chart(fig_usd, use_container_width=True)
        with col_g2:
            st.markdown("##### Harga Minyak Bumi Global (Crude Oil)")
            if "Minyak" in df_ext.columns:
                fig_oil = px.line(df_ext, y="Minyak", color_discrete_sequence=["#EF4444"])
                style_plotly_chart(fig_oil)
                fig_oil.update_layout(height=220, margin=dict(l=40, r=10, t=10, b=10))
                st.plotly_chart(fig_oil, use_container_width=True)
                
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            st.markdown("##### Komoditas Pertanian Dunia")
            ag_cols = [c for c in ["Gandum", "Kedelai", "Jagung"] if c in df_ext.columns]
            if ag_cols:
                fig_ag = px.line(df_ext, y=ag_cols, color_discrete_sequence=["#3B82F6", "#6366F1", "#EC4899"])
                style_plotly_chart(fig_ag)
                fig_ag.update_layout(height=220, margin=dict(l=40, r=10, t=10, b=10))
                st.plotly_chart(fig_ag, use_container_width=True)
        with col_g4:
            st.markdown("##### Curah Hujan Rata-rata (MA7)")
            if "curah_hujan_mm" in df_wea.columns:
                # Show rolling 7-day average of rainfall for smoother display
                df_wea["curah_hujan_ma7"] = df_wea["curah_hujan_mm"].rolling(7).mean()
                fig_rain = px.line(df_wea, y="curah_hujan_ma7", color_discrete_sequence=["#60A5FA"])
                style_plotly_chart(fig_rain)
                fig_rain.update_layout(height=220, margin=dict(l=40, r=10, t=10, b=10))
                st.plotly_chart(fig_rain, use_container_width=True)


# ── Tab 3: Performa Model ──────────────────────────────────────────────────
with tab3:
    results = load_results()
    if results is None:
        render_alert_warning("Hasil evaluasi performa model belum tersedia. Silakan jalankan training pipeline terlebih dahulu.")
    else:
        st.markdown(f"""
        <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 1.25rem; font-family: 'Plus Jakarta Sans', sans-serif;">
            Metrik Evaluasi Performa Model untuk {selected_commodity}
        </h3>
        """, unsafe_allow_html=True)
        
        # Load results for selected commodity
        commodity_results = results.get(selected_commodity, {})
        if not commodity_results:
            # Fallback for old style results.json if it exists
            commodity_results = results
            
        col1, col2, col3 = st.columns(3)
        metrics_list = []
        for name, m in commodity_results.items():
            if isinstance(m, dict) and "MAE" in m:
                metrics_list.append({
                    "Model": name.upper(), 
                    "MAE (Rp)": m["MAE"],
                    "RMSE (Rp)": m["RMSE"], 
                    "MAPE (%)": m["MAPE"]
                })

        if metrics_list:
            df_metrics = pd.DataFrame(metrics_list)

            with col1:
                fig_mae = px.bar(df_metrics, x="Model", y="MAE (Rp)", color="Model",
                                 color_discrete_sequence=["#3B82F6", "#6366F1", "#EC4899"])
                style_plotly_chart(fig_mae)
                fig_mae.update_layout(showlegend=False, height=260, title=dict(text="MAE - Nilai Lebih Rendah Lebih Baik", font=dict(size=12, color="#9CA3AF")))
                fig_mae.update_xaxes(showgrid=False)
                st.plotly_chart(fig_mae, use_container_width=True)
                
            with col2:
                fig_rmse = px.bar(df_metrics, x="Model", y="RMSE (Rp)", color="Model",
                                  color_discrete_sequence=["#3B82F6", "#6366F1", "#EC4899"])
                style_plotly_chart(fig_rmse)
                fig_rmse.update_layout(showlegend=False, height=260, title=dict(text="RMSE - Nilai Lebih Rendah Lebih Baik", font=dict(size=12, color="#9CA3AF")))
                fig_rmse.update_xaxes(showgrid=False)
                st.plotly_chart(fig_rmse, use_container_width=True)
                
            with col3:
                fig_mape = px.bar(df_metrics, x="Model", y="MAPE (%)", color="Model",
                                  color_discrete_sequence=["#3B82F6", "#6366F1", "#EC4899"])
                style_plotly_chart(fig_mape)
                fig_mape.update_layout(showlegend=False, height=260, title=dict(text="MAPE % - Error Persentase Terendah", font=dict(size=12, color="#9CA3AF")))
                fig_mape.update_xaxes(showgrid=False)
                st.plotly_chart(fig_mape, use_container_width=True)

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.dataframe(
                df_metrics.style.format({"MAE (Rp)": "Rp {:,.0f}", "RMSE (Rp)": "Rp {:,.0f}", "MAPE (%)": "{:.2f}%"}),
                use_container_width=True, hide_index=True
            )
            
            best = min(commodity_results, key=lambda k: commodity_results[k]["MAPE"] if (isinstance(commodity_results[k], dict) and "MAPE" in commodity_results[k]) else 99999)
            best_mape = commodity_results[best]['MAPE']
            
            st.markdown(f"""
            <div style="
                background-color: rgba(59, 130, 246, 0.05);
                border: 1px solid rgba(59, 130, 246, 0.25);
                border-radius: 8px;
                padding: 1rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                color: #F9FAFB;
                margin-top: 1.5rem;
            ">
                <div style="color: #3B82F6; flex-shrink: 0; display: flex; align-items: center;">
                    {SVG_TROPHY}
                </div>
                <div style="font-size: 0.875rem; color: #E5E7EB; font-family: 'Inter', sans-serif;">
                    Berdasarkan evaluasi pengujian sekuensial untuk <strong>{selected_commodity}</strong>, model prediktif dengan performa terbaik adalah <strong>{best.upper()}</strong> dengan nilai deviasi error (MAPE) terkecil sebesar <strong>{best_mape:.2f}%</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_alert_warning("Format metrik performa tidak cocok. Silakan jalankan ulang training pipeline.")


# ── Tab 4: Tentang ──────────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
<div style="margin-bottom: 2rem; margin-top: 0.25rem;">
<h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.75rem; font-family: 'Plus Jakarta Sans', sans-serif;">
Deskripsi Proyek Prediksi Pangan Pokok
</h3>
<p style="font-size: 0.9rem; line-height: 1.6; color: #9CA3AF; font-family: 'Inter', sans-serif;">
Sistem ini merupakan platform prediktif harga pangan strategis nasional tingkat eceran di Indonesia. Menggunakan pendekatan Deep Learning berbasis deret waktu multivariate (*Multivariate Time Series Forecasting*), sistem secara otomatis mengintegrasikan indikator fluktuasi pasar domestik serta variabel makroekonomi global untuk menghasilkan proyeksi pergerakan harga komoditas pangan pokok secara andal.
</p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
<div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0; font-size: 0.95rem; color: #F9FAFB; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
<span style="color: #3B82F6; display: flex; align-items: center;">{SVG_INFO}</span>
<span>Integrasi Sumber Data</span>
</h4>
<ul style="list-style: none; padding-left: 0; margin: 0; font-size: 0.85rem; color: #9CA3AF; display: flex; flex-direction: column; gap: 0.75rem; font-family: 'Inter', sans-serif;">
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #3B82F6; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Badan Pangan Nasional (Bapanas / PIHPS)</strong> - Data runtun waktu harga pangan pokok di pasar domestik eceran.</span>
</li>
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #3B82F6; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Bank Indonesia</strong> - Pergerakan fluktuasi kurs mata uang rupiah (USD/IDR) sebagai proksi inflasi impor.</span>
</li>
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #3B82F6; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Yahoo Finance</strong> - Indikator makro komoditas energi global (Crude Oil) serta komoditas pangan dunia (gandum, jagung, kedelai).</span>
</li>
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #3B82F6; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Badan Meteorologi Klimatologi dan Geofisika (BMKG)</strong> - Data curah hujan historis domestik guna mendeteksi gangguan cuaca dan masa panen.</span>
</li>
</ul>
</div>

<div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 1.5rem;">
<h4 style="margin-top: 0; font-size: 0.95rem; color: #F9FAFB; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
<span style="color: #6366F1; display: flex; align-items: center;">{SVG_BRAIN}</span>
<span>Arsitektur Model Komputasi</span>
</h4>
<ul style="list-style: none; padding-left: 0; margin: 0; font-size: 0.85rem; color: #9CA3AF; display: flex; flex-direction: column; gap: 0.75rem; font-family: 'Inter', sans-serif;">
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #6366F1; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Temporal Fusion Transformer (TFT)</strong> - Arsitektur model utama berbasis self-attention dengan kemampuan interpretasi variabel input.</span>
</li>
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #6366F1; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Long Short-Term Memory (LSTM)</strong> - Jaringan saraf rekuren (RNN) standar industri untuk data sekuensial temporal jangka panjang.</span>
</li>
<li style="display: flex; align-items: flex-start; gap: 0.5rem;">
<span style="color: #6366F1; flex-shrink: 0; font-weight: bold; margin-top: -0.1rem;">&check;</span>
<span><strong>Gated Recurrent Unit (GRU)</strong> - Model sekuensial rekuren dengan parameter terkompresi yang memiliki efisiensi komputasi tinggi.</span>
</li>
</ul>
</div>
</div>

<div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1.25rem; display: flex; flex-wrap: wrap; gap: 1.5rem; font-size: 0.8rem; color: #6B7280; justify-content: space-between; font-family: 'Inter', sans-serif;">
<span><strong>Metrik Pengujian:</strong> Mean Absolute Error (MAE) · Root Mean Squared Error (RMSE) · Mean Absolute Percentage Error (MAPE)</span>
<span><strong>Teknologi Komputasi:</strong> Python · PyTorch · Plotly · Streamlit Dark Engine</span>
</div>
""", unsafe_allow_html=True)


# ── Tab 5: Pusat Kontrol Pipeline ──────────────────────────────────────────
with tab5:
    import subprocess
    import time
    
    st.markdown("""
    <h3 style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.75rem; font-family: 'Plus Jakarta Sans', sans-serif;">
        Pusat Kontrol & Pipeline Data
    </h3>
    <p style="font-size: 0.9rem; line-height: 1.6; color: #9CA3AF; font-family: 'Inter', sans-serif; margin-bottom: 1.5rem;">
        Di halaman ini, Anda dapat menjalankan seluruh pipeline secara otomatis. Proses ini akan mengunduh data terbaru dari Yahoo Finance (pasar global), mengambil curah hujan dan hari libur nasional, merapikan data, dan melakukan pelatihan ulang (*retraining*) untuk ketiga arsitektur deep learning (**LSTM**, **GRU**, dan **TFT**).
    </p>
    """, unsafe_allow_html=True)

    # Check process status in session state
    if "pipeline_proc" not in st.session_state:
        st.session_state.pipeline_proc = None

    proc = st.session_state.pipeline_proc
    is_running = False

    if proc is not None:
        poll = proc.poll()
        if poll is None:
            is_running = True
        else:
            # Process finished
            st.session_state.pipeline_proc = None
            if "pipeline_log_file" in st.session_state and st.session_state.pipeline_log_file:
                st.session_state.pipeline_log_file.close()
                st.session_state.pipeline_log_file = None
            
            # Clear caches so new data/models load instantly
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Pipeline selesai dijalankan secara penuh! Semua model dan data baru berhasil dimuat.")
            st.rerun()

    if is_running:
        st.markdown(f"""
        <div style="background-color: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.75rem;">
            <div style="width: 10px; height: 10px; background-color: #3B82F6; border-radius: 50%; animation: pulse 1.5s infinite ease-in-out;"></div>
            <div style="font-size: 0.85rem; color: #E5E7EB; font-family: 'Inter', sans-serif;">
                <strong>Status: Sedang Berjalan...</strong> Model sedang dilatih di latar belakang. Proses ini memerlukan waktu beberapa menit.
            </div>
        </div>
        <style>
            @keyframes pulse {{
                0% {{ transform: scale(0.85); opacity: 0.5; }}
                50% {{ transform: scale(1.15); opacity: 1; }}
                100% {{ transform: scale(0.85); opacity: 0.5; }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Stop process button
        if st.button("Hentikan Proses Training", type="secondary", use_container_width=True):
            proc.terminate()
            proc.wait()
            st.session_state.pipeline_proc = None
            if "pipeline_log_file" in st.session_state and st.session_state.pipeline_log_file:
                st.session_state.pipeline_log_file.close()
                st.session_state.pipeline_log_file = None
            st.warning("Proses training dihentikan secara paksa oleh pengguna.")
            st.rerun()

        # Display logs
        log_path = os.path.join(MODEL_DIR, "pipeline.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                log_text = "".join(lines[-35:])
        else:
            log_text = "Memulai logging..."

        st.markdown(f"""
        <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF; display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
            <span style="color: #9CA3AF; display: inline-flex; align-items: center;">{SVG_SETTINGS}</span>
            <span>Live Logs (35 baris terakhir):</span>
        </span>
        """, unsafe_allow_html=True)
        st.code(log_text, language="text")

        # Autorefresh loop by waiting 2 seconds and rerunning
        time.sleep(2)
        st.rerun()

    else:
        # Display latest logs if they exist
        log_path = os.path.join(MODEL_DIR, "pipeline.log")
        if os.path.exists(log_path):
            st.markdown(f"""
            <span style="font-size: 0.85rem; font-weight: 500; color: #9CA3AF; display: flex; align-items: center; gap: 0.35rem; margin-bottom: 0.5rem; font-family: 'Plus Jakarta Sans', sans-serif;">
                <span style="color: #9CA3AF; display: inline-flex; align-items: center;">{SVG_SETTINGS}</span>
                <span>Log Aktivitas Terakhir:</span>
            </span>
            """, unsafe_allow_html=True)
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                log_text = "".join(lines[-20:])
            st.code(log_text, language="text")

        if st.button("Jalankan Pipeline & Retrain Model", type="primary", use_container_width=True):
            # Reset and prepare log file
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("► Inisialisasi pipeline dari web...\n")
                f.write(f"► Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            log_file = open(log_path, "a", encoding="utf-8")
            
            # Persiapkan env dengan UTF-8 encoding untuk Windows
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            # Start process
            proc = subprocess.Popen(
                [sys.executable, "run_pipeline.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            st.session_state.pipeline_proc = proc
            st.session_state.pipeline_log_file = log_file
            st.rerun()
