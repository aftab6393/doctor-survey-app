from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import joblib
from flask_cors import CORS

# ✅ Initialize Flask & configure templates/static folders
app = Flask(__name__, template_folder="templates", static_folder="static")  
CORS(app)

# ✅ Serve index.html for the frontend
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

        data = request.get_json()  # 🔹 Use get_json() instead of request.json
        if "hour" not in data:
            return jsonify({"error": "Missing 'hour' field in request"}), 400

        input_hour = int(data["hour"])

        # ✅ Load dataset
        file_path = "dummy_npi_data.xlsx"
        df = pd.read_excel(file_path)

        # ✅ Ensure 'Login Hour' exists
        if "Login Hour" not in df.columns:
            df["Login Hour"] = pd.to_datetime(df["Login Time"]).dt.hour

        # ✅ Filter dataset for the input hour
        filtered_df = df[df["Login Hour"] == input_hour]

        # ✅ Load trained model
        model = joblib.load("doctor_prediction_model.pkl")

        # ✅ Make predictions
        X_new = filtered_df[["Login Hour", "Usage Time (mins)", "Speciality", "Region", "Count of Survey Attempts"]]
        predictions = model.predict(X_new)

        filtered_df.loc[:, "Prediction"] = predictions

        # ✅ Select likely doctors
        final_doctors = filtered_df[filtered_df["Prediction"] == 1][["NPI", "State", "Speciality"]]

        # ✅ Save to CSV
        output_file = "filtered_doctors.csv"
        final_doctors.to_csv(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 🔹 Return 500 for server errors

# ✅ Run the app with correct port for Render
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
