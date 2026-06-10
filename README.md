
# 🔥 CalorieAI Pro — Smart Fitness Intelligence Platform

<div align="center">

![Python](<img width="1920" height="1080" alt="Screenshot 2026-06-11 031035" src="https://github.com/user-attachments/assets/3cf4f0ee-6459-4cac-a07a-1fb56fbfc848" />
https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](![Uploading Screenshot 2026-06-11 031114.png…]()
https://img.shields.io/badge/Streamlit-1.46-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](<img width="1920" height="1080" alt="Screenshot 2026-06-11 031146" src="https://github.com/user-attachments/assets/5657687b-89e5-4d49-a468-a61f9b385d6c" />
https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](<img width="1920" height="1080" alt="Screenshot 2026-06-11 031220" src="https://github.com/user-attachments/assets/ce7eff66-941d-4cb0-9bc2-713f7535809f" />
https://img.shields.io/badge/Plotly-6.0-3F4F75?style=for-the-badge&logo=plotly)
![Accuracy](<img width="1920" height="1080" alt="Screenshot 2026-06-11 031256" src="https://github.com/user-attachments/assets/35b3a398-608a-4319-8c6f-6436d25bff74" />
https://img.shields.io/badge/Model%20Accuracy-99.83%25-success?style=for-the-badge)

**An AI-powered calorie burn prediction app built with Random Forest & Streamlit.**

*Created by **RAHUL THAKUR***

</div>

---

## 📌 Overview

**CalorieAI Pro** is a full-stack machine learning web application that predicts how many calories you burn during a workout using a **300-tree Random Forest Regressor** trained on 15,000 workout records with **99.83% R² accuracy**.

This is not just a predictor — it's a complete fitness intelligence platform with 9 feature-rich pages.

## 🌐 Live Demo

🚀 Experience the application live:

[![Launch App](https://img.shields.io/badge/🚀%20Launch-CalorieAI%20Pro-success?style=for-the-badge)](https://calories-ai-pro-gawmaakpc6ebfdtnkmdimz.streamlit.app/)

🔗 **Live URL:** YOUR_STREAMLIT_APP_LINK



---

## ✨ Features at a Glance

| Page | Features |
|------|----------|
| 🏠 **Dashboard** | Live stats, dataset overview, BMI snapshot, recent sessions |
| 🔮 **Predict & Analyze** | AI prediction, workout presets, MET, TDEE %, smart tips, JSON export |
| 📊 **Data Analytics** | 12+ interactive charts — distributions, correlations, heatmaps, violin plots |
| 🏋️ **Workout Planner** | Preset comparisons, weekly plan builder, calorie goal tracker |
| 🍽️ **Nutrition Tracker** | Food log, intake vs burn balance, TDEE comparison, pie charts |
| 📈 **Progress Tracker** | Session history charts, all-time stats, CSV export |
| 🧬 **Body Composition** | BMI, body fat %, lean mass, fat mass, TDEE, ideal weight |
| 🤖 **Model Insights** | Feature importances, actual vs predicted, residuals, R²/MAE/RMSE |
| ⚙️ **Settings** | Profile, weekly goal, data export, session management |

---

## 🗂️ Project Structure

```
calorie-ai-pro/
├── app.py                          # Main Streamlit application (1,287 lines)
├── calorie_prediction_model.pkl    # Trained Random Forest model
├── calories_data.csv               # Dataset (15,000 records)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
# Make sure all 4 files are in the same folder:
# app.py, calorie_prediction_model.pkl, calories_data.csv, requirements.txt
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` 🎉

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Number of Trees | 300 |
| Max Depth | 10 |
| Criterion | Squared Error |
| Training Samples | 15,000 |
| R² Score | **99.83%** |
| Mean Absolute Error | **1.76 kcal** |
| RMSE | **2.59 kcal** |
| Bootstrap | True |

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Binary (0=Male, 1=Female) | Biological sex |
| Age | Integer (10–80) | Age in years |
| Height | Float (cm) | Height in centimetres |
| Weight | Float (kg) | Weight in kilograms |
| Duration | Float (min) | Workout duration |
| Heart_Rate | Float (bpm) | Average heart rate |
| Body_Temp | Float (°C) | Body temperature |

### Feature Importance Ranking

1. 🥇 **Duration** — 91.7% (dominant predictor)
2. 🥈 **Heart Rate** — 4.8%
3. 🥉 **Age** — 2.6%
4. Gender — 0.7%
5. Weight — 0.2%
6. Height — 0.08%
7. Body Temp — 0.02%

---

## 📊 Dataset Overview

- **Records:** 15,000 workout entries
- **Calorie Range:** 1 – 314 kcal
- **Gender Split:** ~50/50 Male/Female
- **Duration Range:** 1 – 90 minutes
- **Key Correlations with Calories:**
  - Duration: 0.955
  - Heart Rate: 0.898
  - Body Temp: 0.825

---

## 🎨 Tech Stack

- **Frontend:** Streamlit 1.46 with custom CSS (glassmorphism dark UI)
- **Charting:** Plotly (12+ chart types: scatter, histogram, violin, heatmap, radar, pie, bar)
- **ML:** scikit-learn Random Forest Regressor
- **Data:** Pandas, NumPy
- **Deployment-ready:** Single command launch

---

## 💡 Smart Features Highlights

- **Workout Presets** — 10 activity types (Running, HIIT, Yoga, Swimming, etc.) with instant predictions
- **Weekly Planner** — Plan 7 days, auto-calculate weekly calorie burn vs goal
- **Nutrition Tracker** — Log meals, compare intake vs burn vs TDEE
- **Body Composition** — BMI, body fat %, lean mass, TDEE, ideal weight
- **Progress Charts** — Session-over-session burn trends
- **Data Explorer** — Filter 15K rows interactively and export to CSV
- **Model Explainability** — Feature importances, actual vs predicted scatter, residuals
- **Export** — JSON prediction export, CSV history export, full session export

---

## 📸 App Pages Preview

```
Dashboard      → Quick stats, overview charts, recent sessions
Predict        → Enter details → instant AI calorie prediction
Analytics      → 12+ interactive charts on the 15K dataset
Planner        → Compare workouts, build weekly schedule
Nutrition      → Food log + net calorie balance
Progress       → Burn history charts over sessions
Body Comp      → BMI, body fat, lean mass breakdown
Model Insights → R², MAE, RMSE, feature importances, residuals
Settings       → Profile, goals, data export
```

---

## 👤 Author

**RAHUL THAKUR**

> *"Building intelligent tools that make fitness science accessible to everyone."*

---

## 📝 License

This project is for educational and portfolio purposes.

---

<div align="center">
  Made with 🔥 by <strong>Rahul Thakur</strong> · CalorieAI Pro v2.0
</div>
