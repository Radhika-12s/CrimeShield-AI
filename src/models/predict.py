import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_model(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = df["Date"].dt.hour
    df["Weekend"] = df["Date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # --- FIX: Handle missing 'Arrest' column ---
    if "Arrest" in df.columns:
        df["RiskLevel"] = df["Arrest"].apply(lambda x: 1 if x == True else 0)
    else:
        # If column missing, default all to 0 (low risk)
        df["RiskLevel"] = 0

    X = df[["Latitude", "Longitude", "Hour", "Weekend"]]
    y = df["RiskLevel"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y)

    return model, scaler


def predict_risk(df, lat, lon, hour, weekend):

    model, scaler = train_model(df)

    input_data = pd.DataFrame({
        "Latitude": [lat],
        "Longitude": [lon],
        "Hour": [hour],
        "Weekend": [weekend]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # You can still calculate CVI
    cvi = 85 if prediction == 1 else 15

    return cvi
