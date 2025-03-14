from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import joblib
import os  # ✅ Added to get Render's assigned port
from flask_cors import CORS

# ✅ Initialize Flask & configure templates/static folders
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ✅ Load trained model and encoders at startup
model = joblib.load("doctor_prediction_model.pkl")
le_speciality = joblib.load("label_encoder_speciality.pkl")
le_region = joblib.load("label_encoder_region.pkl")

# ✅ Serve frontend
@app.route("/")
def home():
    return render_template("index.html")  # Loads index.html from "templates" folder

# ✅ Corrected /predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Ensure request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if "hour" not in data:
            return jsonify({"error": "Missing 'hour' field in request"}), 400

        input_hour = int(data["hour"])

        # ✅ Load dataset (Ensure it's available on Render)
        file_path = "dummy_npi_data.xlsx"
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 500

        df = pd.read_excel(file_path)

        # ✅ Ensure 'Login Hour' exists
        if "Login Hour" not in df.columns:
            df["Login Hour"] = pd.to_datetime(df["Login Time"]).dt.hour

        # ✅ Convert categorical values to numbers
        if "Speciality" in df.columns:
            df["Speciality"] = df["Speciality"].map(
                lambda x: le_speciality.transform([x])[0] if x in le_speciality.classes_ else -1
            )
        if "Region" in df.columns:
            df["Region"] = df["Region"].map(
                lambda x: le_region.transform([x])[0] if x in le_region.classes_ else -1
            )

        # ✅ Filter dataset for the input hour
        filtered_df = df[df["Login Hour"] == input_hour]

        # ✅ Select input features
        features = ["Login Hour", "Usage Time (mins)", "Speciality", "Region", "Count of Survey Attempts"]
        X_new = filtered_df[features]

        # ✅ Ensure there is data to predict
        if X_new.empty:
            return jsonify({"error": "No matching records found for this hour"}), 404

        # ✅ Make predictions
        predictions = model.predict(X_new)
        filtered_df = filtered_df.copy()  # ✅ Fixes SettingWithCopyWarning
        filtered_df.loc[:, "Prediction"] = predictions


        # ✅ Select likely doctors
        final_doctors = filtered_df[filtered_df["Prediction"] == 1][["NPI", "State", "Speciality"]]

        # ✅ Ensure we have predictions
        if final_doctors.empty:
            return jsonify({"message": "No doctors found for this hour"}), 200

        # ✅ Save to CSV
        output_file = "filtered_doctors.csv"
        final_doctors.to_csv(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 🔹 Return 500 for server errors

# ✅ Run the app with correct port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Render dynamically assigns a port
    app.run(debug=True, host="0.0.0.0", port=port)
