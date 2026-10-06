"""Data loading and preprocessing utilities for SmartCard AI."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    "annual_fee", "minimum_income", "credit_score_requirement",
    "cashback_rate", "reward_rate", "travel_reward", "dining_reward",
    "fuel_reward", "shopping_reward", "online_reward", "lounge_access",
    "welcome_bonus", "foreign_transaction_fee"
]

def load_dataset(path="credit_cards.csv"):
    return pd.read_csv(Path(path))

def clean_dataset(df):
    out = df.copy().drop_duplicates()
    numeric_cols = out.select_dtypes(include="number").columns
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    return out

def prepare_classification_data(df):
    clean = clean_dataset(df)
    X = clean[BASE_FEATURES].copy()
    y = clean["card_category"].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return clean, X, y, scaler, X_scaled
