from apiflask import APIFlask, Schema
from marshmallow.fields import String, Integer, Float
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import os

app = APIFlask("TP_API_IA")

class PredictionInput(Schema):
    year = Integer(required=True, metadata={
        "description": "The year of the accident",
        "example": 2023
    })
    vehicle_type = String(required=True, metadata={
        "description": "Type of vehicle",
        "example": "car"
    })

class PredictionOutput(Schema):
    predicted_severity = Float(metadata={
        "description": "Predicted accident severity",
        "example": 10.5
    })
    vehicle_type = String(metadata={
        "description": "Type of vehicle involved in the accident",
        "example": "car"
    })
    year = Integer(metadata={
        "description": "Year of the accident",
        "example": 2023
    })

def load_and_combine_data():
    severity_data = pd.read_csv("data/usagers-2023.csv", sep=";")
    vehicle_data = pd.read_csv("data/vehicules-2023.csv", sep=";")
    combined = pd.merge(severity_data, vehicle_data, on="year")
    return combined

def prepare_training_data():
    data = load_and_combine_data()
    severity_map = {1: 1, 2: 100, 3: 10, 4: 5, 5: 1}
    data['severity'] = data['severity'].map(severity_map)
    X = data[['year', 'vehicle_type']]
    y = data['severity']
    return X, y

def train_and_save_model():
    X, y = prepare_training_data()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['year']),
            ('cat', OneHotEncoder(), ['vehicle_type'])
        ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/accident_model.pkl")

if os.path.exists("models/accident_model.pkl"):
    model = joblib.load("models/accident_model.pkl")
else:
    train_and_save_model()
    model = joblib.load("models/accident_model.pkl")

@app.route("/")
def hello():
    return {
        "message": "Accident Severity Prediction API",
        "status": "ready" if model else "model not loaded",
        "endpoints": {"POST /predict": "Make severity prediction"}
    }

@app.post('/predict')
@app.input(PredictionInput)
@app.output(PredictionOutput)
def predict(json_data):
    if not model:
        return {"error": "Model not available"}, 503
    input_data = pd.DataFrame([{
        'year': json_data['year'],
        'vehicle_type': json_data['vehicle_type']
    }])
    prediction = model.predict(input_data)[0]
    return {
        "predicted_severity": float(prediction),
        "vehicle_type": json_data['vehicle_type'],
        "year": json_data['year']
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)
