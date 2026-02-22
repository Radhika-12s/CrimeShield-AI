import pandas as pd

def load_default_data():
    return pd.read_csv("data/raw/crime_2025_present.csv")

def load_uploaded_data(file):
    return pd.read_csv(file)