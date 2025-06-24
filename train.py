"""Train a car price prediction model."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


DATA_PATH = Path("car_data.csv")
MODEL_PATH = Path("random_forest_regression_model_org.pkl")


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Prepare features and target for training."""
    y = df["Selling_Price"]
    X = df.drop(columns=["Selling_Price"])

    numeric_features = ["Year", "Present_Price", "Kms_Driven", "Owner"]
    categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
    )

    return model, X_train, X_test, y_train, y_test


def train_model() -> None:
    """Train the model and save it to disk."""
    df = load_data(DATA_PATH)
    model, X_train, X_test, y_train, y_test = preprocess(df)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
