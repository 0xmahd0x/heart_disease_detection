from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import joblib

# =========================================
# Load Saved Files
# =========================================

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

results_df = pd.read_csv("models_results.csv")

with open("best_model.txt", "r") as f:
    best_model = f.read()

# =========================================
# Flask App
# =========================================

app = Flask(__name__)

# =========================================
# Home Route
# =========================================

@app.route("/")
def home():

    accuracy = round(results_df["Accuracy"].max() * 100, 2)

    return render_template(
        "index.html",
        best_model=best_model,
        accuracy=accuracy,
        tables=[results_df.to_html(classes='table table-dark table-striped', index=False)],
        prediction_text=None,
        confidence=None
    )

# =========================================
# Prediction Route
# =========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = [

            float(request.form["HighBP"]),
            float(request.form["HighChol"]),
            float(request.form["CholCheck"]),
            float(request.form["BMI"]),
            float(request.form["Smoker"]),
            float(request.form["Stroke"]),
            float(request.form["Diabetes"]),
            float(request.form["PhysActivity"]),
            float(request.form["Fruits"]),
            float(request.form["Veggies"]),
            float(request.form["HvyAlcoholConsump"]),
            float(request.form["AnyHealthcare"]),
            float(request.form["NoDocbcCost"]),
            float(request.form["GenHlth"]),
            float(request.form["MentHlth"]),
            float(request.form["PhysHlth"]),
            float(request.form["DiffWalk"]),
            float(request.form["Sex"]),
            float(request.form["Age"]),
            float(request.form["Education"]),
            float(request.form["Income"])

        ]

        input_array = np.array(input_data).reshape(1, -1)

        scaled_data = scaler.transform(input_array)

        prediction = model.predict(scaled_data)[0]

        confidence = 0

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(scaled_data)

            confidence = round(np.max(probability) * 100, 2)

        if prediction == 1:

            result = "High Risk of Heart Disease"

            result_class = "danger"

        else:

            result = "Low Risk of Heart Disease"

            result_class = "success"

        accuracy = round(results_df["Accuracy"].max() * 100, 2)

        return render_template(
            "index.html",
            prediction_text=result,
            confidence=confidence,
            result_class=result_class,
            best_model=best_model,
            accuracy=accuracy,
            tables=[results_df.to_html(classes='table table-dark table-striped', index=False)]
        )

    except Exception as e:

        accuracy = round(results_df["Accuracy"].max() * 100, 2)

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            confidence=None,
            result_class="warning",
            best_model=best_model,
            accuracy=accuracy,
            tables=[results_df.to_html(classes='table table-dark table-striped', index=False)]
        )

# =========================================
# Run App
# =========================================

if __name__ == "__main__":
    app.run(debug=True)