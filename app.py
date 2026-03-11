import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("StudentsPerformance.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.rename(columns={
    "race/ethnicity":              "ethnicity",
    "parental_level_of_education": "parental_edu",
    "test_preparation_course":     "test_prep",
    "math_score":                  "math",
    "reading_score":               "reading",
    "writing_score":               "writing"
}, inplace=True)

df["reading_writing_avg"] = (df["reading"] + df["writing"]) / 2

le_dict = {}
for col in ["gender", "parental_edu", "lunch", "test_prep"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

model = LinearRegression()
model.fit(df[["gender","parental_edu","lunch","test_prep","reading_writing_avg"]], df["math"])

def predict(gender, parental_edu, lunch, test_prep, reading, writing):
    score = model.predict([[
        le_dict["gender"].transform([gender])[0],
        le_dict["parental_edu"].transform([parental_edu])[0],
        le_dict["lunch"].transform([lunch])[0],
        le_dict["test_prep"].transform([test_prep])[0],
        (reading + writing) / 2
    ]])[0]
    return f"📊 Predicted Math Score: {np.clip(score, 0, 100):.1f} / 100"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["female","male"],                          label="Gender"),
        gr.Dropdown(["high school","some high school",
                     "some college","associate's degree",
                     "bachelor's degree","master's degree"],    label="Parental Education"),
        gr.Dropdown(["standard","free/reduced"],                label="Lunch Type"),
        gr.Dropdown(["completed","none"],                       label="Test Preparation"),
        gr.Slider(0, 100, value=70,                             label="Reading Score"),
        gr.Slider(0, 100, value=70,                             label="Writing Score"),
    ],
    outputs=gr.Textbox(label="📊 Prediction Result"),
    title="🎓 Student Math Score Predictor",
    description="Predicts Math Score using Linear Regression",
).launch()
```

---

### File 3 — `StudentsPerformance.csv`
```
This one you CANNOT type manually — it has 1000 rows!

Instead:
1. Click "Add file" → "Upload files"
2. Drag the CSV from your computer or Colab
