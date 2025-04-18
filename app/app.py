from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import pickle
import os

# Import retraining logic
from app.model import train_model, load_metrics, MODEL_DIR

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

# Load or train initial model
MODEL_PATH = "model/LR_schedule_model.pkl"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    train_model()
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Features processing
def prepare_features(input_data, historical_data):
    input_data["DEMAND_DATE"] = pd.to_datetime(input_data["DEMAND_DATE"], format="%Y-%m")
    input_data["YEAR"] = input_data["DEMAND_DATE"].dt.year
    input_data["MONTH"] = input_data["DEMAND_DATE"].dt.month

    # Historical averages
    prod_avg = historical_data.groupby("PROD_ID")["DEMAND_QUANT"].mean().to_dict()
    input_data["PROD_AVG_DEMAND"] = input_data["PROD_ID"].map(prod_avg)

    # Lag1
    hist_sorted = historical_data.sort_values("DEMAND_DATE")
    lag1 = hist_sorted.groupby("PROD_ID")["DEMAND_QUANT"].last().to_dict()
    input_data["lag1"] = input_data["PROD_ID"].map(lag1)

    # Moving averages
    mv3 = hist_sorted.groupby("PROD_ID")["DEMAND_QUANT"].apply(lambda x: x.tail(3).mean()).to_dict()
    mv6 = hist_sorted.groupby("PROD_ID")["DEMAND_QUANT"].apply(lambda x: x.tail(6).mean()).to_dict()
    input_data["moving_avg_3"] = input_data["PROD_ID"].map(mv3)
    input_data["moving_avg_6"] = input_data["PROD_ID"].map(mv6)

    input_data.ffill(inplace=True)
    features = ["YEAR", "MONTH", "TIRE_SALES", "PROD_AVG_DEMAND", "lag1", "moving_avg_3", "moving_avg_6"]
    return input_data[features]

@app.route("/metrics.json")
def metrics_json():
    metrics_path = os.path.join(os.getcwd(), MODEL_DIR, "metrics.json")
    return send_file(metrics_path, mimetype="application/json")

@app.route("/", methods=["GET"])
def index():
    # Serve the UI
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    month = data.get('month')
    records = data.get('data', [])
    if not month or not records:
        return jsonify({"error": "Invalid payload: 'month' and 'data' required."}), 400

    # Build input DataFrame
    df = pd.DataFrame(records)
    df['DEMAND_DATE'] = pd.to_datetime(month, format="%Y-%m")
    df.rename(columns={'PREVIOUS_MONTH_SALES': 'TIRE_SALES'}, inplace=True)

    # Load historical data
    historical = pd.read_csv("app/HISTORICAL DATA.csv")
    historical["DEMAND_DATE"] = pd.to_datetime(historical["DEMAND_DATE"], format="%Y-%m")

    # Prepare features and predict
    X = prepare_features(df.copy(), historical)
    preds = model.predict(X).round().astype(int)
    df["PREDICTED_DEMAND_QUANT"] = preds

    # Capacity calculation
    prod_times = pd.read_csv("app/PRODUCTION TIMES.csv")
    merged = df.merge(prod_times, on="PROD_ID", how="left")
    merged['TOTAL_TIME'] = merged['PREDICTED_DEMAND_QUANT'] * merged['PROD_TIME']

    total_capacity = 24*60*30
    monthly_time   = merged["TOTAL_TIME"].sum()
    cap_pct        = round(monthly_time/total_capacity*100,2)
    overload_flag  = cap_pct > 100
    alerts         = [month] if overload_flag else []

    # Build response
    response = {
        "month": month,
        "predictions": [
            {
                "PROD_ID": int(r.PROD_ID),
                "TIRE_SALES": int(r.TIRE_SALES),
                "PREDICTED_DEMAND_QUANT": int(p)
            }
            for r, p in zip(df.itertuples(index=False), preds)
        ],
        "capacity": {"value": round(cap_pct,2), "overload": bool(overload_flag)},
        "alerts": [month] if bool(overload_flag) else []
    }
    return jsonify(response)

@app.route("/retrain", methods=["POST"])
def retrain():
    data = request.get_json()
    month = data.get('month')
    records = data.get('data', [])

    if not month or not records:
        return jsonify({"error": "Invalid payload: 'month' and 'data' required."}), 400

    # Construir DataFrame y adaptar columnas al esquema original
    new = pd.DataFrame(records)
    new.rename(columns={"PREVIOUS_MONTH_SALES": "TIRE_SALES"}, inplace=True)
    new["DEMAND_DATE"] = pd.to_datetime(month).strftime("%Y-%m")

    # Leer y actualizar CSV histórico
    hist_path = "app/HISTORICAL DATA.csv"
    hist = pd.read_csv(hist_path)
    hist = pd.concat([hist, new], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    hist.to_csv("app/static/data/HISTORICAL DATA.csv", index=False)

    # Asegurar formato de fecha para el cálculo de ventana
    hist["DEMAND_DATE"] = pd.to_datetime(hist["DEMAND_DATE"], format="%Y-%m")
    window_start = hist["DEMAND_DATE"].min().strftime('%Y-%m')
    window_end   = hist["DEMAND_DATE"].max().strftime('%Y-%m')

    # Retrain model
    new_model = train_model()
    global model
    model = new_model

    metrics = load_metrics()
    return jsonify({
        "status": "ok",
        "new_training_window": {"start": window_start, "end": window_end},
        "metrics": metrics
    })

@app.route("/last-month", methods=["GET"])
def last_month():
    try:
        df = pd.read_csv("app/HISTORICAL DATA.csv")
        df["DEMAND_DATE"] = pd.to_datetime(df["DEMAND_DATE"], format="%Y-%m")
        last_date = df["DEMAND_DATE"].max()
        next_month = (last_date + pd.DateOffset(months=1)).strftime("%Y-%m")
        return jsonify({"last_month": last_date.strftime("%Y-%m"), "next_month": next_month})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)