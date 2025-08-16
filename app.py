import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


model = joblib.load("model/race_ethnicity_rf.pkl")
le = joblib.load("model/label_encoder.pkl")


TRAINING_COLUMNS = [
    'math score',
    'reading score',
    'writing score',
    'gender_male',
    'parental level of education_bachelor\'s degree',
    'parental level of education_high school',
    'parental level of education_master\'s degree',
    'parental level of education_some college',
    'parental level of education_some high school',
    'lunch_standard',
    'test preparation course_none'
]


@app.route("/")
def index():
    """Renders the home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receives user input, preprocesses it, and returns a prediction."""
    try:
        form_data = {
            "math score": int(request.form['math_score']),
            "reading score": int(request.form['reading_score']),
            "writing score": int(request.form['writing_score']),
            "gender": request.form['gender'],
            "parental level of education": request.form['parental_education'],
            "lunch": request.form['lunch'],
            "test preparation course": request.form['test_prep']
        }

        input_df = pd.DataFrame([form_data])

        input_processed = pd.get_dummies(input_df)

        input_aligned = input_processed.reindex(columns=TRAINING_COLUMNS, fill_value=0)

        prediction_encoded = model.predict(input_aligned)[0]

        prediction_label = le.inverse_transform([prediction_encoded])[0]

    except Exception as e:
        error_message = f"An error occurred: {e}"
        return render_template("index.html", error=error_message)

    return render_template("index.html", prediction=prediction_label)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)