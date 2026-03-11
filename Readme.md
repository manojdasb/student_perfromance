# 🎓 Student Math Score Prediction

> A Machine Learning project that predicts students' **math scores** based on demographic and academic features — built with Linear Regression and deployed as a live Gradio app.

🌐 **Live Demo** → [Try the App](https://huggingface.co/spaces/manojdasb/student-performance)  
📂 **Dataset** → StudentsPerformance.csv (1000 students)

---

## 📌 Project Overview

This project answers one key question:

> **"What factors affect a student's math score — and can we predict it?"**

We explore how gender, parental education, lunch type, and test preparation influence academic performance, then build an ML model to predict math scores with **87.9% accuracy (R²)**.

---

## 📊 Dataset

| Feature | Type | Example Values |
|---|---|---|
| gender | Categorical | female, male |
| race/ethnicity | Categorical | group A, B, C, D, E |
| parental_level_of_education | Categorical | high school → master's degree |
| lunch | Categorical | standard, free/reduced |
| test_preparation_course | Categorical | completed, none |
| math_score | Numerical ⭐ Target | 0 – 100 |
| reading_score | Numerical | 0 – 100 |
| writing_score | Numerical | 0 – 100 |

**Shape:** 1000 rows × 8 columns | **Missing values:** 0

### Target Variable Stats
```
Mean    →  66.09
Std     →  15.16
Min     →   0.00
Max     → 100.00
Median  →  66.00
```

---

## ⚙️ Feature Engineering

```python
# New feature created from existing ones
reading_writing_avg = (reading_score + writing_score) / 2
```

This single engineered feature became the **strongest predictor** with importance = **13.98**, boosting R² from ~0.35 → **0.879**.

---

## 🤖 Models Trained

| Model | RMSE ↓ | R² ↑ |
|---|---|---|
| ✅ **Linear Regression** | **5.437** | **0.8785** |
| Gradient Boosting | 5.885 | 0.8577 |
| Random Forest | 6.689 | 0.8161 |

### ✅ Best Model: Linear Regression
- **RMSE = 5.437** → predictions off by only ~5.4 marks on average
- **R² = 0.879** → explains **87.9%** of variance in math scores
- Linear relationships between features and target → no need for complex models

---

## 🔑 Key Insights

### Factors Affecting Academic Performance

| Rank | Factor | Score Gap | Best Group | Worst Group |
|---|---|---|---|---|
| 🥇 1st | Parental Education | **10.5 pts** | Master's (73.6) | High School (63.1) |
| 🥈 2nd | Lunch Type | **8.6 pts** | Standard (70.8) | Free/Reduced (62.2) |
| 🥉 3rd | Test Preparation | **7.6 pts** | Completed (72.7) | None (65.0) |
| 4th | Gender | **3.8 pts** | Female (69.6) | Male (65.8) |

### 💡 Takeaways
- **Socioeconomic background** (parental education + lunch type) has the **biggest impact** on scores
- **Test preparation** is the most **actionable factor** — students can actually do something about it
- **Gender** has the **least impact** among all four factors
- Students who score well in **reading & writing** also tend to score well in **math**

---

## 📈 Score Breakdown by Factor

```
Parental Education:
  Master's degree    → 73.6  ████████████████
  Bachelor's degree  → 70.9  ███████████████
  Associate's degree → 68.3  ██████████████
  Some College       → 67.2  ██████████████
  Some High School   → 65.3  █████████████
  High School        → 63.1  █████████████

Lunch Type:
  Standard           → 70.8  ███████████████
  Free / Reduced     → 62.2  █████████████

Test Preparation:
  Completed          → 72.7  ███████████████
  None               → 65.0  █████████████

Gender:
  Female             → 69.6  ██████████████
  Male               → 65.8  █████████████
```

---

## 🗂️ Project Structure

```
student-performance/
│
├── app.py                       ← Gradio web app (loads model.pkl)
├── model.pkl                    ← Saved Linear Regression model
├── encoders.pkl                 ← Saved LabelEncoders
├── StudentsPerformance.csv      ← Dataset (1000 students)
├── requirements.txt             ← Python dependencies
├── student_performance.ipynb    ← Full analysis notebook
└── README.md
```

---

## 🚀 Run Locally

### Step 1 — Clone repo
```bash
git clone https://github.com/manojdasb/student-performance.git
cd student-performance
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run app
```bash
python app.py
```

Open → `http://localhost:7860`

---

## 🖥️ App Preview

```
🎓 Student Math Score Predictor
┌──────────────────────────────────┐
│ Gender          [ female     ▼ ] │
│ Parental Edu    [ master's   ▼ ] │
│ Lunch Type      [ standard   ▼ ] │
│ Test Prep       [ completed  ▼ ] │
│ Reading Score   [═══════70═════] │
│ Writing Score   [═══════70═════] │
│                                  │
│          [ Submit ]              │
└──────────────────────────────────┘

  📊 Predicted Math Score: 78.5 / 100
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML models |
| Matplotlib / Seaborn | Visualizations |
| Gradio | Web app UI |
| Joblib | Model saving/loading |
| Hugging Face Spaces | Deployment |

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
gradio
matplotlib
seaborn
joblib
```

---

## 🌐 Live Demo

👉 **[Try the App on Hugging Face](https://huggingface.co/spaces/manojdas23/student-performance)**

No installation needed — open the link and predict instantly!

---

## 👨‍💻 Author

**Manoj Das B**
- GitHub → [github.com/manojdasb](https://github.com/manojdasb)
- Hugging Face → [huggingface.co/spaces/manojdasb](https://huggingface.co/spaces/manojdas23/student-performance)
