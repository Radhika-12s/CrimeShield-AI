import joblib
import pandas as pd

model = joblib.load("models/crime_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_risk(lat, lon, hour, weekend):
    input_df = pd.DataFrame([[lat, lon, hour, weekend]],
                            columns=["Latitude","Longitude","Hour","Weekend"])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability