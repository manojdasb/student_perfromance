import gradio as gr
import numpy as np
import joblib

# Load saved model & encoders
model   = joblib.load("model.pkl")
le_dict = joblib.load("encoders.pkl")

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
