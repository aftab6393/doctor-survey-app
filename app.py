from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
from flask_cors import CORS

# ✅ Initialize Flask app (Corrected position)
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_hour = int(data["hour"])

        # Load dataset
        file_path = "dummy_npi_data.xlsx"
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name='Dataset')

        # Ensure 'Login Hour' exists
        if "Login Hour" not in df.columns:
            df["Login Hour"] = pd.to_datetime(df["Login Time"]).dt.hour

        # Load trained label encoders
        le_speciality = joblib.load("label_encoder_speciality.pkl")
        le_region = joblib.load("label_encoder_region.pkl")

        # Convert categorical columns into numbers (if they exist)
        if "Speciality" in df.columns:
            df.loc[:, "Speciality"] = df["Speciality"].apply(
                lambda x: le_speciality.transform([x])[0] if x in le_speciality.classes_ else -1)
        if "Region" in df.columns:
            df.loc[:, "Region"] = df["Region"].apply(
                lambda x: le_region.transform([x])[0] if x in le_region.classes_ else -1)

        # Filter dataset for the input hour
        filtered_df = df[df["Login Hour"] == input_hour]

        # Select input features
        features = ["Login Hour", "Usage Time (mins)", "Speciality", "Region", "Count of Survey Attempts"]
        X_new = filtered_df[features]

        # Load trained model
        model = joblib.load("doctor_prediction_model.pkl")

        # Make predictions
        predictions = model.predict(X_new)

        # ✅ Fix Indentation Error
        filtered_df.loc[:, "Prediction"] = predictions  

        # Select likely doctors
        final_doctors = filtered_df[filtered_df["Prediction"] == 1][["NPI", "State", "Speciality"]]

        # Save to CSV
        output_file = "filtered_doctors.csv"
        final_doctors.to_csv(output_file, index=False)from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ✅ Add a Home Route to Fix 404 Error
@app.route("/")
def home():
    return "Welcome to the Doctor Survey Prediction API! Use /predict"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        data = request.json
        input_hour = int(data["hour"])

        # Load dataset
        file_path = "dummy_npi_data.xlsx"
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name='Dataset')

        # Ensure 'Login Hour' exists
        if "Login Hour" not in df.columns:
            df["Login Hour"] = pd.to_datetime(df["Login Time"]).dt.hour

        # Load trained label encoders
        le_speciality = joblib.load("label_encoder_speciality.pkl")
        le_region = joblib.load("label_encoder_region.pkl")

        # Convert categorical columns into numbers (if they exist)
        if "Speciality" in df.columns:
            df.loc[:, "Speciality"] = df["Speciality"].apply(
                lambda x: le_speciality.transform([x])[0] if x in le_speciality.classes_ else -1
            )
        if "Region" in df.columns:
            df.loc[:, "Region"] = df["Region"].apply(
                lambda x: le_region.transform([x])[0] if x in le_region.classes_ else -1
            )

        # Filter dataset for the input hour
        filtered_df = df[df["Login Hour"] == input_hour]

        # Select input features
        features = ["Login Hour", "Usage Time (mins)", "Speciality", "Region", "Count of Survey Attempts"]
        X_new = filtered_df[features]

        # Load trained model
        model = joblib.load("doctor_prediction_model.pkl")

        # Make predictions
        predictions = model.predict(X_new)
        filtered_df.loc[:, "Prediction"] = predictions

        # Select likely available doctors
        final_doctors = filtered_df[filtered_df["Prediction"] == 1][["NPI", "State", "Speciality"]]

        # Save to CSV
        output_file = "filtered_doctors.csv"
        final_doctors.to_csv(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Run the Flask app on the correct host for Render deployment
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)


        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
