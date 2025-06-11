from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('signal_strength_model.pkl')

class SignalInput(BaseModel):
    Frequency: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Precipitation: float
    Altitude_m: float

app = FastAPI()

@app.post('/predict')
def predict(data: SignalInput):
    # Create DataFrame with exact feature names as in training data
    input_df = pd.DataFrame([{
        'Frequency': data.Frequency,
        'Temperature': data.Temperature,
        'Humidity': data.Humidity,
        'Wind Speed': data.Wind_Speed,       # note: space in column name
        'Precipitation': data.Precipitation,
        'Altitude(m)': data.Altitude_m       # note: parentheses in column name
    }])

    prediction = model.predict(input_df)[0]
    return {"predicted_signal_strength": prediction}
