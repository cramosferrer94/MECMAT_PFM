# app/routes.py

from flask import Blueprint, request, jsonify
from app.model import get_trained_model, predict_from_features

main = Blueprint('main', __name__)

# Cargar el modelo entrenado (se leerá el archivo "model.pkl")
trained_model = get_trained_model()

@main.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para recibir nuevos datos y retornar la predicción.
    Se espera recibir un JSON con las siguientes claves:
      - YEAR
      - MONTH
      - TIRE_SALES
      - PROD_AVG_DEMAND
      - lag1
      - moving_avg_3
      - moving_avg_6
    """
    data = request.get_json()
    try:
        input_features = {
            "YEAR": data["YEAR"],
            "MONTH": data["MONTH"],
            "TIRE_SALES": data["TIRE_SALES"],
            "PROD_AVG_DEMAND": data["PROD_AVG_DEMAND"],
            "lag1": data["lag1"],
            "moving_avg_3": data["moving_avg_3"],
            "moving_avg_6": data["moving_avg_6"]
        }
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {str(e)}"}), 400

    prediction = predict_from_features(input_features, trained_model)
    return jsonify({"prediccion": prediction})
