from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("house_model.pkl")

@app.route("/")
def index():
    return {"message": "🏠 House Appraisal System is running"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        bedrooms = data.get("bedrooms", 0)
        bathrooms = data.get("bathrooms", 0)
        sqft = data.get("sqft", 0)

        # Prepare features
        features = np.array([[bedrooms, bathrooms, sqft]])
        predicted_price = model.predict(features)[0]

        return jsonify({"predicted_price": round(float(predicted_price), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
