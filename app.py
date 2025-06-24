"""Flask application for car price prediction."""

import datetime
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request


MODEL_PATH = Path("random_forest_regression_model_org.pkl")


def load_model():
    """Load the pre-trained model from disk."""
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


app = Flask(__name__)
model = load_model()


@app.route("/", methods=["GET"])
def home():
    """Render the main page."""
    return render_template("car/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict car selling price based on form data."""
    form = request.form
    present_price = float(form["Present_Price"])
    year = int(form["Year"])
    kms_driven = int(form["kms_driven"])
    owner = int(form["Owner"])

    fuel_type_petrol = form["Fuel_Type_Petrol"] == "Petrol"
    fuel_type_diesel = 0 if fuel_type_petrol else 1
    fuel_type_petrol = int(fuel_type_petrol)

    current_year = datetime.date.today().year
    year = current_year - year

    seller_type_individual = int(form["Seller_Type_Individual"] == "Individual")
    transmission_manual = int(form["Transmission_Mannual"] == "Manual")

    features = [
        present_price,
        kms_driven,
        owner,
        year,
        fuel_type_diesel,
        fuel_type_petrol,
        seller_type_individual,
        transmission_manual,
    ]

    prediction = model.predict([features])
    output = round(prediction[0], 2)

    if output < 0:
        return render_template("car/index.html", prediction_texts="Sorry you cannot sell this car")

    return render_template(
        "car/index.html",
        prediction_text=f"You Can Sell The Car at {output}",
    )


if __name__ == "__main__":
    app.run(debug=True)
