from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime

# Initialize the FastAPI app
app = FastAPI()

# hello world endpoint
@app.get("/")
def read_root(): # function that is binded with the endpoint
    return {"Hello": "Price Prediction Model API is Working!"}

# Load the trained model from the .h5 file
model = tf.keras.models.load_model('single_layer_nn.keras')

# Load the dataset
national_df = pd.read_csv('national_df_cleaned.csv')

# Extract relevant features for the multivariate model
features = national_df[['gr1_high_price', 'gr1_avg_price', 'gr2_high_price', 'gr2_avg_price', 'white_high_price', 'white_avg_price']]

# Load the time information
time = pd.to_datetime(national_df['published_at'])  # Ensure time is in datetime format

# Scale the features (Note: make sure to use the same scaler as used during training)
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Define the window size used during training
window_size = 20  # Update this to match the window size used during model training

# Function to predict the value for the given future date
def predict_for_date_and_price(date, price_column):
    # Convert the date to datetime format
    target_date = pd.to_datetime(date)

    # Get the index of the desired price column
    price_index = features.columns.get_loc(price_column)

    # Start predictions from the last available point in the dataset
    current_window = features_scaled[-window_size:]
    last_date = time.iloc[-1]

    # Continue predicting step-by-step until reaching the target date
    while last_date < target_date:
        # Predict the next value using the current window
        window_data = np.expand_dims(current_window, axis=0)  # Expand dims to match model input shape
        prediction_scaled = model.predict(window_data)

        # Reshape prediction to match the feature dimension
        prediction_scaled = np.repeat(prediction_scaled, current_window.shape[1], axis=-1)

        # Update the window for the next prediction
        current_window = np.concatenate((current_window[1:], prediction_scaled), axis=0)

        # Update the date
        last_date += pd.Timedelta(weeks=1)  # Assuming weekly predictions; adjust accordingly

    # Inverse transform to original scale and return the prediction for the target date and price column
    prediction = scaler.inverse_transform(prediction_scaled)
    return float(prediction[0][price_index])

# Define the request body model using Pydantic
class PredictionRequest(BaseModel):
    date: str
    price_column: str

# Define the API endpoint for making predictions
@app.post("/predict_price")
async def predict_price(request: PredictionRequest):
    try:
        # Extract date and price column from request body
        date = request.date
        price_column = request.price_column

        # Validate inputs
        if not date or not price_column:
            raise HTTPException(status_code=400, detail="Missing required parameters: date and price_column")

        # Make the prediction
        predicted_value = predict_for_date_and_price(date, price_column)
        return {"predicted_value": predicted_value}

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
