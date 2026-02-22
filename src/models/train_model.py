import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/raw/crime_2025_present.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Hour"] = df["Date"].dt.hour
df["Weekend"] = df["Date"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

df["RiskLevel"] = df["Arrest"].apply(lambda x: 1 if x == True else 0)

X = df[["Latitude", "Longitude", "Hour", "Weekend"]]
y = df["RiskLevel"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

joblib.dump(model, "models/crime_risk_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")