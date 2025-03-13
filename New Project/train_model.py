import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = "dummy_npi_data.xlsx"  # Ensure this is in your working directory
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name='Dataset')

# Convert timestamps to datetime format
df["Login Time"] = pd.to_datetime(df["Login Time"])
df["Logout Time"] = pd.to_datetime(df["Logout Time"])

# Extract login hour
df["Login Hour"] = df["Login Time"].dt.hour

# Create target variable (1 = likely to attempt survey, 0 = not likely)
df["Likely to Attempt"] = (df["Count of Survey Attempts"] > 0).astype(int)

# Encode categorical variables
le_speciality = LabelEncoder()
df["Speciality"] = le_speciality.fit_transform(df["Speciality"])

le_region = LabelEncoder()
df["Region"] = le_region.fit_transform(df["Region"])

# Select features and target
X = df[["Login Hour", "Usage Time (mins)", "Speciality", "Region", "Count of Survey Attempts"]]
y = df["Likely to Attempt"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "doctor_prediction_model.pkl")
joblib.dump(le_speciality, "label_encoder_speciality.pkl")
joblib.dump(le_region, "label_encoder_region.pkl")

print("Model training completed successfully!")
