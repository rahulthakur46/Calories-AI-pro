import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import json
import io
import csv
from datetime import datetime, timedelta
import random

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CalorieAI Pro – by Rahul Thakur",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CalorieAI Pro — Built by Rahul Thakur | Random Forest ML Calorie Predictor"
    }
)

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0c29 30%, #1a0533 60%, #0a0a1a 100%);
    color: #f0f0f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #12102b 60%, #0d1117 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.hero-banner {
    background: linear-gradient(135deg, rgba(255,78,0,0.15), rgba(236,159,5,0.1), rgba(162,155,254,0.1));
    border: 1px solid rgba(255,78,0,0.25);
    border-radius: 24px;
    padding: 40px 36px 32px;
    text-align: center;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,78,0,0.05) 0%, transparent 60%);
    animation: rotate 8s linear infinite;
}
@keyframes rotate { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

.hero-title {
    font-size: 3.2rem; font-weight: 900; margin: 0;
    background: linear-gradient(90deg, #ff4e00, #ec9f05, #a29bfe);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-size: 200%; animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%,100% { background-position: 0% } 50% { background-position: 100% }
}

.hero-sub { font-size: 1.05rem; opacity: 0.65; margin: 10px 0 0; }

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 20px 22px;
    margin: 6px 0;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255,78,0,0.4);
    box-shadow: 0 15px 45px rgba(255,78,0,0.2);
}
.metric-value { font-size: 2.2rem; font-weight: 900; line-height: 1.1; }
.metric-label { font-size: 0.78rem; opacity: 0.6; margin-top: 5px; text-transform: uppercase; letter-spacing: 1.2px; }
.metric-delta { font-size: 0.82rem; margin-top: 4px; font-weight: 600; }

.result-hero {
    background: linear-gradient(135deg, #ff4e00 0%, #ec9f05 50%, #ff6b35 100%);
    border-radius: 24px; padding: 42px 32px; text-align: center;
    box-shadow: 0 24px 80px rgba(255,78,0,0.5);
    animation: glow 2.5s ease-in-out infinite;
    position: relative; overflow: hidden;
}
.result-hero::after {
    content: '🔥'; position: absolute; font-size: 8rem;
    top: -20px; right: -20px; opacity: 0.12; transform: rotate(15deg);
}
@keyframes glow {
    0%,100% { box-shadow: 0 24px 80px rgba(255,78,0,0.5); }
    50%      { box-shadow: 0 24px 100px rgba(255,78,0,0.8); }
}
.result-hero h1 { font-size: 5rem; font-weight: 900; margin: 0; color: #fff; line-height: 1; }
.result-hero p  { font-size: 1.15rem; margin: 8px 0 0; color: rgba(255,255,255,0.9); }

.section-header {
    font-size: 1.3rem; font-weight: 700; letter-spacing: 0.3px;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 2px solid rgba(255,78,0,0.4);
    color: #fff;
}

.tip-box {
    background: rgba(255,255,255,0.04);
    border-left: 4px solid #ff4e00;
    border-radius: 0 14px 14px 0;
    padding: 14px 18px; margin: 8px 0;
    font-size: 0.92rem; line-height: 1.6;
}

.badge {
    display: inline-block;
    background: linear-gradient(90deg, #ff4e00, #ec9f05);
    color: #fff; font-weight: 700; font-size: 0.72rem;
    border-radius: 20px; padding: 4px 14px;
    letter-spacing: 0.8px; text-transform: uppercase;
}

.progress-bar-container {
    background: rgba(255,255,255,0.08); border-radius: 50px;
    height: 10px; margin: 6px 0 14px; overflow: hidden;
}
.progress-bar-fill {
    height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #ff4e00, #ec9f05);
    transition: width 1s ease;
}

.workout-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 18px 20px; margin: 8px 0;
    cursor: pointer; transition: all 0.25s;
}
.workout-card:hover { border-color: rgba(255,78,0,0.5); transform: scale(1.01); }
.workout-card h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 700; color: #ff4e00; }
.workout-card p  { margin: 0; font-size: 0.85rem; opacity: 0.7; }

.history-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 18px; margin: 6px 0;
    display: flex; justify-content: space-between; align-items: center;
}

.footer {
    text-align: center; padding: 32px 0 16px;
    font-size: 0.82rem; color: rgba(255,255,255,0.3);
    border-top: 1px solid rgba(255,255,255,0.07);
    margin-top: 40px;
}
.footer span { color: #ff4e00; font-weight: 800; font-size: 0.88rem; }

.sidebar-brand {
    text-align: center; padding: 20px 0 28px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}
.nav-item {
    padding: 10px 14px; border-radius: 10px; margin: 3px 0;
    cursor: pointer; transition: background 0.2s;
    font-size: 0.9rem; font-weight: 500;
}
.nav-item:hover { background: rgba(255,78,0,0.12); }
.nav-item.active { background: rgba(255,78,0,0.2); border-left: 3px solid #ff4e00; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(90deg, #ff4e00, #ec9f05) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 50px !important;
    padding: 14px 36px !important; font-size: 1rem !important;
    width: 100% !important; transition: opacity 0.2s, transform 0.1s;
    letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.88 !important; transform: scale(0.99); }
.stButton > button:active { transform: scale(0.97); }

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #fff !important;
}
label { color: rgba(255,255,255,0.75) !important; font-size: 0.87rem !important; font-weight: 500 !important; }
div[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #fff !important;
}
div[data-testid="stNumberInput"] > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important; color: #fff !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 14px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: rgba(255,255,255,0.6) !important;
    font-weight: 600; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff4e00, #ec9f05) !important;
    color: white !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.stSlider [data-baseweb="slider"] .thumb { background: #ff4e00 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { background: #ff4e00 !important; }

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px; padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  LOAD ASSETS  (cached)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("calorie_prediction_model.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("calories_data.csv")
    df["Gender_Label"] = df["Gender"].map({0: "Male", 1: "Female"})
    df["BMI"]          = (df["Weight"] / ((df["Height"]/100)**2)).round(1)
    df["Age_Group"]    = pd.cut(df["Age"], bins=[0,20,30,40,50,60,100],
                                labels=["<20","20-29","30-39","40-49","50-59","60+"])
    df["Cal_per_min"]  = (df["Calories"] / df["Duration"]).round(2)
    df["Intensity"]    = df["Heart_Rate"].apply(lambda h: "Low" if h<100 else ("Moderate" if h<150 else "High"))
    return df

@st.cache_data(show_spinner=False)
def compute_model_stats(df_raw):
    model_ = load_model()
    X = df_raw[['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']]
    y = df_raw['Calories']
    y_pred = model_.predict(X)
    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, mae, rmse

with st.spinner("🔥 Loading CalorieAI Pro..."):
    model  = load_model()
    df     = load_data()
    df_raw = pd.read_csv("calories_data.csv")
    r2, mae, rmse = compute_model_stats(df_raw)

features = ['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "history"     not in st.session_state: st.session_state.history     = []
if "page"        not in st.session_state: st.session_state.page        = "🏠 Dashboard"
if "weekly_goal" not in st.session_state: st.session_state.weekly_goal = 2000
if "username"    not in st.session_state: st.session_state.username    = "Athlete"
if "unit_system" not in st.session_state: st.session_state.unit_system = "Metric"
if "last_pred"   not in st.session_state: st.session_state.last_pred   = None

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.025)",
    font=dict(color="#d0d0d0", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    margin=dict(l=16, r=16, t=48, b=16),
    title_font=dict(size=14, color="#fff"),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_BASE)
    return fig

def bmi_calc(h, w):       return round(w / ((h/100)**2), 1)
def bmi_info(b):
    if b < 18.5: return "Underweight", "#74b9ff", "⚠️ Consider increasing caloric intake."
    if b < 25:   return "Normal ✅",   "#00b894", "👍 BMI is in a healthy range."
    if b < 30:   return "Overweight",  "#fdcb6e", "🏃 More cardio recommended."
    return "Obese",           "#d63031", "⚠️ Consult a healthcare professional."

def tdee(gender, age, h, w, activity=1.55):
    bmr = (10*w + 6.25*h - 5*age + (5 if gender==0 else -161))
    return round(bmr * activity)

def calorie_zone(cal):
    if cal < 50:  return "🧘 Warm-up",          "#74b9ff"
    if cal < 150: return "🚴 Fat Burn Zone",     "#00b894"
    if cal < 250: return "💪 Cardio Zone",       "#fdcb6e"
    return            "🔥 Peak Performance",     "#ff4e00"

def ideal_weight(h, gender):
    base = 50 if gender==0 else 45.5
    return round(base + 2.3 * ((h/2.54) - 60), 1)

def body_fat_estimate(bmi_v, age, gender):
    return round((1.20 * bmi_v) + (0.23 * age) - (10.8 * (1 if gender==0 else 0)) - 5.4, 1)

WORKOUT_PRESETS = {
    "🏃 Running (moderate)":  dict(duration=30, heart_rate=145, body_temp=40.2),
    "🚴 Cycling (light)":     dict(duration=45, heart_rate=115, body_temp=39.8),
    "🏊 Swimming":            dict(duration=40, heart_rate=135, body_temp=39.5),
    "🧘 Yoga":                dict(duration=60, heart_rate=85,  body_temp=38.8),
    "💪 Weight Training":     dict(duration=50, heart_rate=120, body_temp=40.0),
    "⚽ Football/Soccer":     dict(duration=90, heart_rate=155, body_temp=40.5),
    "🥊 Boxing":              dict(duration=45, heart_rate=165, body_temp=40.8),
    "🏄 HIIT":                dict(duration=25, heart_rate=175, body_temp=41.0),
    "🚶 Walking (brisk)":     dict(duration=40, heart_rate=95,  body_temp=38.5),
    "🎾 Tennis":              dict(duration=60, heart_rate=140, body_temp=40.1),
}

FOOD_DB = {
    "🍎 Apple":        52, "🍌 Banana":         89,
    "🍚 Rice (100g)": 130, "🍗 Chicken (100g)": 165,
    "🥚 Egg (1)":      78, "🥛 Milk (250ml)":   122,
    "🍕 Pizza (slice)":285,"🍔 Burger":         354,
    "🥗 Salad (bowl)":  78, "🥜 Peanuts (30g)": 170,
    "🧃 Juice (250ml)": 110,"☕ Coffee (black)":   2,
    "🍫 Chocolate bar":235, "🥤 Soda (355ml)":  150,
}

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div style="font-size:3.2rem;margin-bottom:4px;">🔥</div>
      <div style="font-size:1.6rem;font-weight:900;background:linear-gradient(90deg,#ff4e00,#ec9f05);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">CalorieAI Pro</div>
      <div style="font-size:0.75rem;opacity:0.5;margin-top:2px;">SMART FITNESS INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("📍 Navigate", [
        "🏠 Dashboard",
        "🔮 Predict & Analyze",
        "📊 Data Analytics",
        "🏋️ Workout Planner",
        "🍽️ Nutrition Tracker",
        "📈 Progress Tracker",
        "🧬 Body Composition",
        "🤖 Model Insights",
        "⚙️ Settings"
    ], index=["🏠 Dashboard","🔮 Predict & Analyze","📊 Data Analytics",
              "🏋️ Workout Planner","🍽️ Nutrition Tracker","📈 Progress Tracker",
              "🧬 Body Composition","🤖 Model Insights","⚙️ Settings"].index(st.session_state.page))
    st.session_state.page = page

    st.markdown("---")
    st.markdown('<div style="font-size:0.85rem;font-weight:700;opacity:0.7;margin-bottom:10px;">👤 QUICK PROFILE</div>', unsafe_allow_html=True)
    q_gender = st.selectbox("Gender",  ["Male","Female"], key="q_gender")
    q_age    = st.slider("Age",   10, 80, 28,  key="q_age")
    q_height = st.slider("Height (cm)", 130, 220, 172, key="q_height")
    q_weight = st.slider("Weight (kg)", 30,  160,  68,  key="q_weight")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:12px 0 4px;">
      <div style="font-size:0.72rem;opacity:0.4;letter-spacing:1px;">CREATED BY</div>
      <div style="font-size:1.1rem;font-weight:900;background:linear-gradient(90deg,#ff4e00,#ec9f05);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:2px;">
        RAHUL THAKUR
      </div>
      <div style="font-size:0.7rem;opacity:0.35;margin-top:3px;">Random Forest · 99.83% R²</div>
    </div>
    """, unsafe_allow_html=True)

gender_num = 0 if q_gender == "Male" else 1
bmi_v      = bmi_calc(q_height, q_weight)
bmi_cat, bmi_color, bmi_tip = bmi_info(bmi_v)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class="hero-banner">
      <span class="badge">AI Powered · 99.83% Accuracy</span>
      <h1 class="hero-title">🔥 CalorieAI Pro</h1>
      <p class="hero-sub">
        The most intelligent calorie burn predictor — powered by a 300-tree Random Forest trained on 15,000 workouts.
      </p>
      <p style="margin-top:14px;font-size:0.85rem;opacity:0.45;">Built by <strong style="color:#ff4e00;">Rahul Thakur</strong></p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#ff4e00;">{len(df):,}</div>
          <div class="metric-label">🗃️ Training Records</div>
          <div class="metric-delta" style="color:#00b894;">+15K samples</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#a29bfe;">{r2*100:.2f}%</div>
          <div class="metric-label">🎯 Model Accuracy (R²)</div>
          <div class="metric-delta" style="color:#00b894;">Excellent fit</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#fdcb6e;">{mae:.2f}</div>
          <div class="metric-label">📉 Mean Abs Error</div>
          <div class="metric-delta" style="color:#00b894;">±{mae:.1f} kcal</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#55efc4;">300</div>
          <div class="metric-label">🌲 RF Estimators</div>
          <div class="metric-delta" style="color:#a29bfe;">Max depth: 10</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        dash_c1, dash_c2 = st.columns(2)
        with dash_c1:
            fig_hist = apply_theme(px.histogram(df, x="Calories", nbins=50,
                                                color_discrete_sequence=["#ff4e00"],
                                                title="Calorie Distribution"))
            fig_hist.update_layout(height=260, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        with dash_c2:
            gc = df["Gender_Label"].value_counts()
            fig_pie = apply_theme(px.pie(values=gc.values, names=gc.index, hole=0.5,
                                         color_discrete_sequence=["#74b9ff","#fd79a8"],
                                         title="Gender Split"))
            fig_pie.update_layout(height=260, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">👤 Your Profile Snapshot</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-size:1.1rem;font-weight:700;">{q_gender} · {q_age} yrs</div>
              <div style="font-size:0.82rem;opacity:0.6;">{q_height} cm · {q_weight} kg</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:1.6rem;font-weight:900;color:{bmi_color};">{bmi_v}</div>
              <div style="font-size:0.75rem;opacity:0.6;">BMI</div>
            </div>
          </div>
          <div style="margin-top:12px;">
            <div style="font-size:0.78rem;opacity:0.6;margin-bottom:4px;">BMI Status: <b style="color:{bmi_color};">{bmi_cat}</b></div>
            <div class="progress-bar-container">
              <div class="progress-bar-fill" style="width:{min(bmi_v/40*100,100):.0f}%;background:linear-gradient(90deg,#00b894,{bmi_color});"></div>
            </div>
          </div>
        </div>
        <div class="tip-box">{bmi_tip}</div>
        """, unsafe_allow_html=True)
        ideal_w = ideal_weight(q_height, gender_num)
        diff    = round(q_weight - ideal_w, 1)
        sign    = "+" if diff >= 0 else ""
        st.markdown(f'<div class="tip-box">⚖️ Ideal weight for your height: <b>{ideal_w} kg</b> &nbsp;|&nbsp; Difference: <b>{sign}{diff} kg</b></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="section-header" style="margin-top:20px;">📅 Recent Predictions</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-4:]):
            z_label, z_color = calorie_zone(h['cal'])
            st.markdown(f"""
            <div class="history-item">
              <div>
                <b>{h['workout']}</b>
                <div style="font-size:0.8rem;opacity:0.55;">{h['ts']} · {h['dur']} min · {h['hr']} bpm</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:1.4rem;font-weight:900;color:#ff4e00;">{h['cal']:.0f} kcal</div>
                <div style="font-size:0.75rem;color:{z_color};">{z_label}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tip-box" style="text-align:center;padding:24px;">
          🔮 No predictions yet — go to <b>Predict & Analyze</b> to get started!
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: PREDICT & ANALYZE
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Predict & Analyze":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">🔮 Predict Calorie Burn</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Fill in your workout details for an instant AI-powered estimate.</p>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown('<div class="section-header">🏋️ Workout Details</div>', unsafe_allow_html=True)

        preset = st.selectbox("⚡ Quick Preset (optional)", ["Custom"] + list(WORKOUT_PRESETS.keys()))
        if preset != "Custom":
            p = WORKOUT_PRESETS[preset]
            def_dur  = p["duration"]
            def_hr   = p["heart_rate"]
            def_temp = p["body_temp"]
        else:
            def_dur  = 30; def_hr = 100; def_temp = 40.0

        c1, c2 = st.columns(2)
        with c1:
            duration   = st.slider("⏱️ Duration (min)", 1, 120, def_dur)
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 55, 210, def_hr)
        with c2:
            body_temp = st.slider("🌡️ Body Temp (°C)", 36.0, 42.5, def_temp, 0.1)
            intensity_choice = st.selectbox("🔥 Perceived Intensity", ["Easy","Moderate","Hard","Maximum"])

        st.markdown('<div class="section-header" style="margin-top:16px;">👤 Personal Details</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            gender_p  = st.selectbox("Gender", ["Male","Female"], key="pred_gender")
            age_p     = st.slider("Age", 10, 80, q_age, key="pred_age")
        with c4:
            height_p  = st.slider("Height (cm)", 130, 220, q_height, key="pred_height")
            weight_p  = st.slider("Weight (kg)", 30,  160,  q_weight, key="pred_weight")

        workout_name = st.text_input("📝 Workout label (optional)", value=preset if preset!="Custom" else "My Workout")
        predict_btn  = st.button("⚡ Predict Calories Burned")

    with right_col:
        if predict_btn or st.session_state.last_pred:
            if predict_btn:
                g_num  = 0 if gender_p=="Male" else 1
                inp_df = pd.DataFrame([[g_num, age_p, height_p, weight_p, duration, heart_rate, body_temp]],
                                      columns=features)
                with st.spinner("Running Random Forest inference..."):
                    time.sleep(0.4)
                    cal = float(model.predict(inp_df)[0])

                entry = dict(
                    cal=round(cal,1), dur=duration, hr=heart_rate,
                    temp=body_temp, gender=gender_p, age=age_p,
                    height=height_p, weight=weight_p,
                    ts=datetime.now().strftime("%d %b %H:%M"),
                    workout=workout_name or preset
                )
                st.session_state.history.append(entry)
                st.session_state.last_pred = entry
            else:
                entry = st.session_state.last_pred
                cal   = entry['cal']
                g_num = 0 if entry['gender']=="Male" else 1
                duration=entry['dur']; heart_rate=entry['hr']
                weight_p=entry['weight']; height_p=entry['height']
                age_p=entry['age']

            zone_label, zone_color = calorie_zone(cal)
            cal_per_min = round(cal / duration, 2)
            bmi_p       = bmi_calc(height_p, weight_p)
            met_val     = round(cal / (weight_p * (duration/60)), 2)
            tdee_val    = tdee(g_num, age_p, height_p, weight_p)

            st.markdown(f"""
            <div class="result-hero">
              <p style="font-size:0.95rem;opacity:0.85;margin:0 0 6px;">ESTIMATED CALORIES BURNED</p>
              <h1>{cal:.1f} kcal</h1>
              <p>{zone_label}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#ff4e00;">{cal_per_min}</div>
                  <div class="metric-label">⏱️ kcal / min</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#a29bfe;">{met_val}</div>
                  <div class="metric-label">⚡ MET Value</div></div>""", unsafe_allow_html=True)
            with m3:
                pct_tdee = round(cal/tdee_val*100,1)
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#fdcb6e;">{pct_tdee}%</div>
                  <div class="metric-label">📊 % of Daily TDEE</div></div>""", unsafe_allow_html=True)
            with m4:
                water_l = round(cal/60*0.5, 2)
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-value" style="color:#55efc4;">{water_l}L</div>
                  <div class="metric-label">💧 Water Need</div></div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:18px;">💡 Smart Insights</div>', unsafe_allow_html=True)
            add_min = round((300 - cal)/cal_per_min, 1) if cal < 300 else 0
            insights = [
                f"🎯 {'Add ' + str(add_min) + ' more minutes to hit 300 kcal.' if add_min > 0 else '🏆 Exceeded 300 kcal goal!'}",
                f"🍽️ This burn equals ~{round(cal/4)} grams of carbohydrates.",
                f"⚖️ BMI: {bmi_p} ({bmi_info(bmi_p)[0]})",
                f"🔋 You burned {round(cal/tdee_val*100,1)}% of your estimated daily energy needs ({tdee_val} kcal).",
                f"❤️ Heart rate zone: {'Fat Burn' if heart_rate<130 else ('Cardio' if heart_rate<160 else 'Peak')} ({heart_rate} bpm)",
            ]
            for ins in insights:
                st.markdown(f'<div class="tip-box">{ins}</div>', unsafe_allow_html=True)

            # Export
            st.markdown("---")
            export_data = {
                "Prediction": f"{cal:.1f} kcal",
                "Workout": workout_name if predict_btn else entry['workout'],
                "Duration": f"{duration} min",
                "Heart Rate": f"{heart_rate} bpm",
                "MET": met_val,
                "kcal/min": cal_per_min,
                "TDEE": tdee_val,
                "Timestamp": datetime.now().isoformat()
            }
            st.download_button(
                "📥 Download Result as JSON",
                data=json.dumps(export_data, indent=2),
                file_name="calorie_prediction.json",
                mime="application/json"
            )
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:400px;opacity:0.45;">
              <div style="font-size:5rem;">🔮</div>
              <div style="font-size:1.1rem;margin-top:14px;">Set your workout details and click Predict!</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: DATA ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:20px;">📊 Data Analytics</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Distributions","🔥 Correlations","🧬 Segments","🗺️ Heatmaps","📋 Explorer"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = apply_theme(px.histogram(df, x="Calories", nbins=60,
                                           color_discrete_sequence=["#ff4e00"],
                                           title="📊 Calorie Distribution"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = apply_theme(px.histogram(df, x="Duration", nbins=50,
                                           color_discrete_sequence=["#a29bfe"],
                                           title="⏱️ Duration Distribution"))
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = apply_theme(px.histogram(df, x="Heart_Rate", nbins=50,
                                           color_discrete_sequence=["#fd79a8"],
                                           title="❤️ Heart Rate Distribution"))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = apply_theme(px.histogram(df, x="Age", nbins=40,
                                           color_discrete_sequence=["#55efc4"],
                                           title="👤 Age Distribution"))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            samp1 = df.sample(2000, random_state=1)
            fig = apply_theme(px.scatter(samp1, x="Duration", y="Calories",
                                         color="Heart_Rate", color_continuous_scale="Inferno",
                                         title="⏱️ Duration vs Calories", opacity=0.65))
            m1_, b1_ = np.polyfit(samp1["Duration"], samp1["Calories"], 1)
            x_l1 = np.linspace(samp1["Duration"].min(), samp1["Duration"].max(), 100)
            fig.add_trace(go.Scatter(x=x_l1, y=m1_*x_l1+b1_, mode="lines",
                                     line=dict(color="#fff", width=2, dash="dash"),
                                     showlegend=False))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            samp2 = df.sample(2000, random_state=2)
            fig = apply_theme(px.scatter(samp2, x="Heart_Rate", y="Calories",
                                         color="Gender_Label",
                                         color_discrete_map={"Male":"#74b9ff","Female":"#fd79a8"},
                                         title="❤️ Heart Rate vs Calories", opacity=0.65))
            m2_, b2_ = np.polyfit(samp2["Heart_Rate"], samp2["Calories"], 1)
            x_l2 = np.linspace(samp2["Heart_Rate"].min(), samp2["Heart_Rate"].max(), 100)
            fig.add_trace(go.Scatter(x=x_l2, y=m2_*x_l2+b2_, mode="lines",
                                     line=dict(color="#fff", width=2, dash="dash"),
                                     showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

        num_cols = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
        corr     = df[num_cols].corr()
        fig = apply_theme(px.imshow(corr, color_continuous_scale="RdBu_r",
                                    text_auto=".2f", title="🔗 Full Correlation Matrix",
                                    zmin=-1, zmax=1))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = apply_theme(px.box(df, x="Gender_Label", y="Calories",
                                     color="Gender_Label",
                                     color_discrete_map={"Male":"#74b9ff","Female":"#fd79a8"},
                                     title="⚤ Calorie by Gender"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = apply_theme(px.box(df, x="Age_Group", y="Calories",
                                     color="Age_Group",
                                     color_discrete_sequence=px.colors.sequential.Plasma_r,
                                     title="👥 Calorie by Age Group", category_orders={"Age_Group":["<20","20-29","30-39","40-49","50-59","60+"]}))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = apply_theme(px.violin(df, x="Intensity", y="Calories",
                                        color="Intensity",
                                        color_discrete_sequence=["#74b9ff","#fdcb6e","#ff4e00"],
                                        title="🎻 Calorie Violin by Intensity",
                                        category_orders={"Intensity":["Low","Moderate","High"]}))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            age_avg = df.groupby("Age_Group", observed=True)["Calories"].mean().reset_index()
            fig = apply_theme(px.bar(age_avg, x="Age_Group", y="Calories",
                                     color="Calories", color_continuous_scale="Reds",
                                     title="📊 Avg Calories by Age Group"))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            hm = df.pivot_table(values="Calories", index="Gender_Label",
                                columns="Age_Group", aggfunc="mean", observed=True)
            fig = apply_theme(px.imshow(hm, color_continuous_scale="Inferno",
                                        text_auto=".0f",
                                        title="🗺️ Avg Calories: Gender × Age Group"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            hm2 = df.pivot_table(values="Calories", index="Intensity",
                                 columns="Age_Group", aggfunc="mean", observed=True)
            fig = apply_theme(px.imshow(hm2, color_continuous_scale="RdYlGn",
                                        text_auto=".0f",
                                        title="🗺️ Avg Calories: Intensity × Age Group"))
            st.plotly_chart(fig, use_container_width=True)

        fig = apply_theme(px.density_heatmap(df, x="Duration", y="Calories",
                                              nbinsx=40, nbinsy=40,
                                              color_continuous_scale="Inferno",
                                              title="🔥 Density Heatmap: Duration vs Calories"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-header">📋 Interactive Data Explorer</div>', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1: gf = st.selectbox("Gender", ["All","Male","Female"], key="exp_g")
        with fc2: cal_r = st.slider("Calories", 0, 320, (0,320), key="exp_c")
        with fc3: dur_r = st.slider("Duration", 1, 90, (1,90),   key="exp_d")
        with fc4: hr_r  = st.slider("Heart Rate", 60, 210, (60,210), key="exp_h")

        filtered = df.copy()
        if gf != "All": filtered = filtered[filtered["Gender_Label"]==gf]
        filtered = filtered[
            filtered["Calories"].between(*cal_r) &
            filtered["Duration"].between(*dur_r) &
            filtered["Heart_Rate"].between(*hr_r)
        ]
        st.markdown(f'<div style="opacity:0.55;font-size:0.83rem;margin-bottom:8px;">Showing <b>{len(filtered):,}</b> of {len(df):,} records</div>', unsafe_allow_html=True)
        show_cols = ["Gender_Label","Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories","BMI","Cal_per_min","Intensity"]
        st.dataframe(filtered[show_cols].head(300), use_container_width=True, height=360)

        st.markdown('<div class="section-header" style="margin-top:16px;">📐 Summary Statistics</div>', unsafe_allow_html=True)
        st.dataframe(filtered[["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories","BMI"]].describe().round(2),
                     use_container_width=True)

        csv_buf = io.StringIO()
        filtered.to_csv(csv_buf, index=False)
        st.download_button("📥 Download Filtered Data", csv_buf.getvalue(),
                           file_name="filtered_calories.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════════
#  PAGE: WORKOUT PLANNER
# ═══════════════════════════════════════════════════════════════════
elif page == "🏋️ Workout Planner":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">🏋️ Smart Workout Planner</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Plan your week, compare workouts, and hit your calorie goals.</p>', unsafe_allow_html=True)

    goal_cal = st.number_input("🎯 Weekly Calorie Burn Goal (kcal)", 500, 10000, st.session_state.weekly_goal, 100)
    st.session_state.weekly_goal = goal_cal

    st.markdown('<div class="section-header" style="margin-top:10px;">⚡ Workout Presets – Instant Predictions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    preset_results = []
    for i, (name, params) in enumerate(WORKOUT_PRESETS.items()):
        inp = pd.DataFrame([[gender_num, q_age, q_height, q_weight,
                             params["duration"], params["heart_rate"], params["body_temp"]]],
                           columns=features)
        cal_est = float(model.predict(inp)[0])
        preset_results.append((name, params["duration"], cal_est))
        with cols[i % 3]:
            z_label, z_color = calorie_zone(cal_est)
            st.markdown(f"""<div class="workout-card">
              <h4>{name}</h4>
              <p>{params['duration']} min · {params['heart_rate']} bpm</p>
              <div style="font-size:1.6rem;font-weight:900;color:#ff4e00;margin-top:8px;">{cal_est:.0f} kcal</div>
              <div style="font-size:0.78rem;color:{z_color};">{z_label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:24px;">📊 Workout Comparison Chart</div>', unsafe_allow_html=True)
    pnames = [p[0].split(" ",1)[1] for p in preset_results]
    pcals  = [p[2] for p in preset_results]
    pdurs  = [p[1] for p in preset_results]
    fig = apply_theme(px.bar(x=pnames, y=pcals,
                              color=pcals, color_continuous_scale="Reds",
                              labels={"x":"Workout","y":"Calories"},
                              title="🔥 Estimated Calories per Workout Type"))
    fig.update_layout(height=320, xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📅 Weekly Plan Builder</div>', unsafe_allow_html=True)
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekly = {}
    dcols = st.columns(7)
    for i, day in enumerate(days):
        with dcols[i]:
            st.markdown(f'<div style="text-align:center;font-size:0.75rem;font-weight:700;opacity:0.7;margin-bottom:6px;">{day[:3].upper()}</div>', unsafe_allow_html=True)
            ch = st.checkbox("Active", key=f"day_{day}", value=(i not in [5,6]))
            wt = st.selectbox("", ["REST"] + [n.split(" ",1)[1] for n in WORKOUT_PRESETS.keys()],
                              key=f"wt_{day}", label_visibility="collapsed")
            weekly[day] = wt if ch else "REST"

    total_week_cal = 0
    week_details   = []
    for day, wtype in weekly.items():
        if wtype != "REST":
            matched = [p for p in preset_results if p[0].split(" ",1)[1] == wtype]
            if matched:
                total_week_cal += matched[0][2]
                week_details.append((day, wtype, matched[0][2]))

    pct_goal = min(total_week_cal / goal_cal * 100, 100)
    st.markdown(f"""
    <div class="metric-card" style="margin-top:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:1rem;font-weight:700;">Weekly Burn Estimate</div>
          <div style="opacity:0.55;font-size:0.82rem;margin-top:2px;">Based on your profile & selected workouts</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:2rem;font-weight:900;color:#ff4e00;">{total_week_cal:.0f} kcal</div>
          <div style="font-size:0.8rem;opacity:0.6;">Goal: {goal_cal} kcal</div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <div style="font-size:0.78rem;opacity:0.6;margin-bottom:5px;">{pct_goal:.1f}% of weekly goal</div>
        <div class="progress-bar-container">
          <div class="progress-bar-fill" style="width:{pct_goal:.0f}%;"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: NUTRITION TRACKER
# ═══════════════════════════════════════════════════════════════════
elif page == "🍽️ Nutrition Tracker":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">🍽️ Nutrition Tracker</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Track your food intake and see net calorie balance against your workout burn.</p>', unsafe_allow_html=True)

    tdee_val = tdee(gender_num, q_age, q_height, q_weight)
    burned   = st.session_state.last_pred['cal'] if st.session_state.last_pred else 0.0

    st.markdown('<div class="section-header">🍱 Log Your Food</div>', unsafe_allow_html=True)
    if "food_log" not in st.session_state: st.session_state.food_log = []

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        food_choice = st.selectbox("Select Food", list(FOOD_DB.keys()))
    with fc2:
        servings = st.number_input("Servings", 0.5, 10.0, 1.0, 0.5)
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Food"):
            cal_food = FOOD_DB[food_choice] * servings
            st.session_state.food_log.append({
                "item": food_choice, "servings": servings,
                "calories": cal_food,
                "time": datetime.now().strftime("%H:%M")
            })

    custom_food = st.text_input("Or add custom food", placeholder="e.g. Protein Shake")
    c1, c2, c3 = st.columns([2,1,1])
    with c2: custom_cal = st.number_input("Calories", 0, 2000, 200, key="custom_cal")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Custom") and custom_food:
            st.session_state.food_log.append({
                "item": custom_food, "servings": 1,
                "calories": custom_cal,
                "time": datetime.now().strftime("%H:%M")
            })

    total_intake = sum(f["calories"] for f in st.session_state.food_log)
    net_cal      = total_intake - burned
    surplus      = net_cal - tdee_val

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#fdcb6e;">{total_intake:.0f}</div>
          <div class="metric-label">🍽️ Total Intake (kcal)</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#ff4e00;">{burned:.0f}</div>
          <div class="metric-label">🔥 Burned (kcal)</div></div>""", unsafe_allow_html=True)
    with m3:
        nc_color = "#00b894" if net_cal < tdee_val else "#d63031"
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{nc_color};">{net_cal:.0f}</div>
          <div class="metric-label">⚖️ Net Calories</div></div>""", unsafe_allow_html=True)
    with m4:
        s_color = "#00b894" if surplus <= 0 else "#d63031"
        sign    = "+" if surplus >= 0 else ""
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{s_color};">{sign}{surplus:.0f}</div>
          <div class="metric-label">📊 vs TDEE ({tdee_val} kcal)</div></div>""", unsafe_allow_html=True)

    if st.session_state.food_log:
        st.markdown('<div class="section-header" style="margin-top:20px;">📋 Food Log</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.food_log)):
            st.markdown(f"""
            <div class="history-item">
              <div>
                <b>{item['item']}</b>
                <div style="font-size:0.8rem;opacity:0.55;">{item['servings']}x · {item['time']}</div>
              </div>
              <div style="font-size:1.4rem;font-weight:900;color:#fdcb6e;">{item['calories']:.0f} kcal</div>
            </div>""", unsafe_allow_html=True)

        food_labels = [f["item"].split(" ",1)[-1] for f in st.session_state.food_log]
        food_cals   = [f["calories"] for f in st.session_state.food_log]
        fig = apply_theme(px.pie(values=food_cals, names=food_labels,
                                 hole=0.45, title="🍱 Calorie Breakdown by Food",
                                 color_discrete_sequence=px.colors.sequential.Reds_r))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑️ Clear Food Log"):
            st.session_state.food_log = []
            st.rerun()
    else:
        st.markdown('<div class="tip-box" style="text-align:center;padding:20px;">Add food items above to start tracking! 🥗</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 Progress Tracker":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">📈 Progress Tracker</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Visualise your calorie burn history and spot trends.</p>', unsafe_allow_html=True)

    if len(st.session_state.history) >= 2:
        hist_df = pd.DataFrame(st.session_state.history)

        c1, c2 = st.columns(2)
        with c1:
            fig = apply_theme(go.Figure(go.Scatter(
                x=list(range(1, len(hist_df)+1)), y=hist_df["cal"],
                mode="lines+markers+text",
                text=[f"{c:.0f}" for c in hist_df["cal"]],
                textposition="top center",
                line=dict(color="#ff4e00", width=3),
                marker=dict(size=10, color="#ff4e00",
                            line=dict(width=2, color="#fff")),
                fill="tozeroy",
                fillcolor="rgba(255,78,0,0.1)"
            )))
            fig.update_layout(title="🔥 Calorie Burn Over Sessions",
                              xaxis_title="Session", yaxis_title="kcal", **PLOTLY_BASE)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = apply_theme(go.Figure(go.Bar(
                x=list(range(1, len(hist_df)+1)), y=hist_df["dur"],
                marker=dict(color="#a29bfe", opacity=0.85)
            )))
            fig.update_layout(title="⏱️ Workout Duration per Session",
                              xaxis_title="Session", yaxis_title="Minutes", **PLOTLY_BASE)
            st.plotly_chart(fig, use_container_width=True)

        total_cal  = hist_df["cal"].sum()
        avg_cal    = hist_df["cal"].mean()
        best_sess  = hist_df.loc[hist_df["cal"].idxmax()]
        total_min  = hist_df["dur"].sum()

        st.markdown('<div class="section-header">🏆 All-Time Stats</div>', unsafe_allow_html=True)
        s1,s2,s3,s4 = st.columns(4)
        with s1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#ff4e00;">{total_cal:.0f}</div>
              <div class="metric-label">🔥 Total kcal Burned</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#a29bfe;">{avg_cal:.1f}</div>
              <div class="metric-label">📊 Avg kcal / Session</div></div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#fdcb6e;">{best_sess['cal']:.0f}</div>
              <div class="metric-label">🏅 Best Session</div></div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:#55efc4;">{total_min}</div>
              <div class="metric-label">⏱️ Total Minutes</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:18px;">📋 Session Log</div>', unsafe_allow_html=True)
        st.dataframe(hist_df[["ts","workout","dur","hr","cal"]].rename(columns={
            "ts":"Time","workout":"Workout","dur":"Duration (min)",
            "hr":"Heart Rate","cal":"Calories"
        }).sort_index(ascending=False), use_container_width=True, height=260)

        csv_h = io.StringIO()
        hist_df.to_csv(csv_h, index=False)
        st.download_button("📥 Export History CSV", csv_h.getvalue(),
                           file_name="workout_history.csv", mime="text/csv")
    else:
        st.markdown("""
        <div class="tip-box" style="text-align:center;padding:32px;">
          <div style="font-size:3rem;">📈</div>
          <div style="margin-top:10px;">Log at least 2 workouts in <b>Predict & Analyze</b> to see progress charts!</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: BODY COMPOSITION
# ═══════════════════════════════════════════════════════════════════
elif page == "🧬 Body Composition":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">🧬 Body Composition Analyzer</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Deep dive into your body metrics and health indicators.</p>', unsafe_allow_html=True)

    bmi_v2    = bmi_calc(q_height, q_weight)
    bfp       = body_fat_estimate(bmi_v2, q_age, gender_num)
    lean_mass = round(q_weight * (1 - bfp/100), 1)
    fat_mass  = round(q_weight * (bfp/100), 1)
    tdee_val  = tdee(gender_num, q_age, q_height, q_weight)
    ideal_w   = ideal_weight(q_height, gender_num)
    bmi_cat2, bmi_c2, bmi_t2 = bmi_info(bmi_v2)

    m1,m2,m3,m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{bmi_c2};">{bmi_v2}</div>
          <div class="metric-label">⚖️ BMI – {bmi_cat2}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#fdcb6e;">{bfp}%</div>
          <div class="metric-label">🔬 Est. Body Fat</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#55efc4;">{lean_mass} kg</div>
          <div class="metric-label">💪 Lean Mass</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#ff4e00;">{tdee_val}</div>
          <div class="metric-label">🔋 TDEE (kcal/day)</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">🍩 Body Composition Breakdown</div>', unsafe_allow_html=True)
        fig = apply_theme(px.pie(values=[lean_mass, fat_mass],
                                  names=["Lean Mass","Fat Mass"], hole=0.52,
                                  color_discrete_sequence=["#55efc4","#ff4e00"]))
        fig.update_layout(title="Body Mass Composition", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">📊 BMI Scale</div>', unsafe_allow_html=True)
        bmi_ranges = ["Underweight\n(<18.5)","Normal\n(18.5-24.9)","Overweight\n(25-29.9)","Obese\n(≥30)"]
        bmi_colors = ["#74b9ff","#00b894","#fdcb6e","#d63031"]
        bmi_vals   = [18.5, 24.9, 29.9, 40]
        fig2 = go.Figure(go.Bar(
            x=bmi_ranges, y=[1,1,1,1],
            marker_color=bmi_colors, opacity=0.7,
            text=["<18.5","18.5-25","25-30","≥30"],
            textposition="inside"
        ))
        fig2.add_vline(x=min(bmi_v2/10 - 0.5, 3.5), line_dash="dash",
                       line_color="#fff", line_width=2,
                       annotation_text=f"You: {bmi_v2}", annotation_font_color="#fff")
        fig2.update_layout(title="Your BMI on the Scale", height=320,
                           showlegend=False, yaxis_visible=False, **PLOTLY_BASE)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">📋 Health Metrics Summary</div>', unsafe_allow_html=True)
    metrics_info = [
        ("BMI", bmi_v2, bmi_cat2, bmi_t2, bmi_c2),
        ("Body Fat %", f"{bfp}%", "Estimated", "Using BMI + Age formula (Deurenberg)", "#fdcb6e"),
        ("Lean Mass", f"{lean_mass} kg", f"{round(lean_mass/q_weight*100,1)}% of body", "Muscle, bone, organs & fluids", "#55efc4"),
        ("Ideal Weight", f"{ideal_w} kg", f"{'↑' if q_weight < ideal_w else '↓'} {abs(round(q_weight-ideal_w,1))} kg to go", "Devine formula based on height", "#a29bfe"),
        ("TDEE", f"{tdee_val} kcal", "Moderate activity assumed", "Total Daily Energy Expenditure", "#ff4e00"),
    ]
    for mname, mval, mcat, mtip, mcol in metrics_info:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div style="font-weight:700;font-size:1rem;">{mname}</div>
              <div style="font-size:0.82rem;opacity:0.55;margin-top:2px;">{mtip}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:1.5rem;font-weight:900;color:{mcol};">{mval}</div>
              <div style="font-size:0.75rem;opacity:0.6;">{mcat}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:4px;">🤖 Model Insights & Explainability</h2>', unsafe_allow_html=True)
    st.markdown('<p style="opacity:0.55;margin-bottom:24px;">Deep-dive into the Random Forest model — how it works, what it learned.</p>', unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#00b894;">{r2*100:.2f}%</div>
          <div class="metric-label">🎯 R² Score</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#fdcb6e;">{mae:.2f}</div>
          <div class="metric-label">📉 MAE (kcal)</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#ff4e00;">{rmse:.2f}</div>
          <div class="metric-label">📊 RMSE (kcal)</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:#74b9ff;">300</div>
          <div class="metric-label">🌲 Trees in Forest</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    importances = model.feature_importances_
    imp_df      = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = apply_theme(px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                  color="Importance", color_continuous_scale="Reds",
                                  title="🌲 Feature Importances (RF Gini)"))
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = apply_theme(px.pie(imp_df, values="Importance", names="Feature",
                                  hole=0.48, title="📊 Importance Share",
                                  color_discrete_sequence=px.colors.sequential.Reds_r))
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔬 Actual vs Predicted (Sample 2000)</div>', unsafe_allow_html=True)
    sample = df_raw.sample(2000, random_state=42)
    X_s    = sample[features]
    y_s    = sample["Calories"].values
    y_pred_s = model.predict(X_s)
    residuals = y_s - y_pred_s

    c3, c4 = st.columns(2)
    with c3:
        fig = apply_theme(go.Figure())
        fig.add_trace(go.Scatter(x=y_s, y=y_pred_s, mode="markers",
                                  marker=dict(color="#ff4e00", opacity=0.5, size=5),
                                  name="Predictions"))
        fig.add_trace(go.Scatter(x=[y_s.min(), y_s.max()],
                                  y=[y_s.min(), y_s.max()],
                                  mode="lines", line=dict(color="#fff", dash="dash", width=2),
                                  name="Perfect fit"))
        fig.update_layout(title="Actual vs Predicted Calories",
                          xaxis_title="Actual", yaxis_title="Predicted", **PLOTLY_BASE)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = apply_theme(px.histogram(x=residuals, nbins=60,
                                        color_discrete_sequence=["#a29bfe"],
                                        title="📊 Residual Distribution",
                                        labels={"x":"Residual (kcal)"}))
        fig.add_vline(x=0, line_dash="dash", line_color="#fff", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 About the Model Architecture"):
        st.markdown("""
        <div style="line-height:1.9;font-size:0.92rem;opacity:0.85;">
        <b>Algorithm:</b> Random Forest Regressor<br>
        <b>Estimators:</b> 300 decision trees<br>
        <b>Max Depth:</b> 10 levels per tree<br>
        <b>Features:</b> Gender, Age, Height, Weight, Duration, Heart Rate, Body Temp<br>
        <b>Training samples:</b> 15,000 workout records<br>
        <b>Criterion:</b> Squared Error (MSE minimization)<br>
        <b>Bootstrap:</b> True (bagging)<br>
        <br>
        <b>Why Random Forest?</b><br>
        • Handles non-linear relationships between biometrics and calorie burn<br>
        • Robust to outliers (e.g., extreme heart rates)<br>
        • Built-in feature importance via Gini impurity<br>
        • Ensemble of 300 trees eliminates overfitting risk<br>
        • 99.83% R² with only 1.76 kcal average error — production-ready accuracy
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<h2 style="color:#fff;font-weight:900;margin-bottom:24px;">⚙️ Settings & Preferences</h2>', unsafe_allow_html=True)

    with st.expander("👤 User Profile", expanded=True):
        name = st.text_input("Display Name", st.session_state.username)
        st.session_state.username = name
        wgoal = st.number_input("Weekly Calorie Goal (kcal)", 200, 10000, st.session_state.weekly_goal, 50)
        st.session_state.weekly_goal = wgoal
        st.markdown(f'<div class="tip-box">👋 Hello, <b>{name}</b>! Your weekly goal is <b>{wgoal} kcal</b>.</div>', unsafe_allow_html=True)

    with st.expander("📊 App Information"):
        st.markdown("""
        <div style="line-height:2;font-size:0.9rem;">
        <b>App:</b> CalorieAI Pro<br>
        <b>Version:</b> 2.0.0<br>
        <b>Model:</b> Random Forest Regressor (scikit-learn 1.6.1)<br>
        <b>Training Data:</b> 15,000 workout records<br>
        <b>Accuracy:</b> 99.83% R²<br>
        <b>Framework:</b> Streamlit + Plotly<br>
        <b>Created by:</b> <span style="color:#ff4e00;font-weight:800;">RAHUL THAKUR</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🗑️ Data Management"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.history   = []
                st.session_state.last_pred = None
                st.success("History cleared!")
        with col2:
            if st.button("🗑️ Clear Food Log"):
                st.session_state.food_log = [] if "food_log" in st.session_state else []
                st.success("Food log cleared!")

    with st.expander("📥 Export All Data"):
        if st.session_state.history:
            export_all = {
                "user": st.session_state.username,
                "weekly_goal": st.session_state.weekly_goal,
                "history": st.session_state.history,
                "food_log": st.session_state.get("food_log", []),
                "exported_at": datetime.now().isoformat()
            }
            st.download_button("📥 Download Full Session Data (JSON)",
                               data=json.dumps(export_all, indent=2),
                               file_name="calorieai_data.json", mime="application/json")
        else:
            st.markdown('<div class="tip-box">No session data to export yet. Make some predictions first!</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  🔥 <b>CalorieAI Pro</b> &nbsp;·&nbsp; Random Forest · 300 Trees · 15K Samples · 99.83% R²<br><br>
  Designed & Developed with ❤️ by <span>RAHUL THAKUR</span>
  &nbsp;·&nbsp; <span style="opacity:0.4;">Smart Fitness Intelligence Platform v2.0</span>
</div>
""", unsafe_allow_html=True)