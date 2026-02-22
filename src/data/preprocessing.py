import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Keep only necessary columns
    required_cols = ["Latitude", "Longitude", "Date"]
    df = df[[col for col in df.columns if col in df.columns]]

    # Convert Date to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Hour"] = df["Date"].dt.hour
        df["Weekend"] = df["Date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Drop rows without coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    return df