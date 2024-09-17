from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import pandas as pd
from src.utils import load_and_preprocess_data, list_datasets
from src.model import train_model

app = FastAPI()

# Serve static files (like HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

class DatasetSelection(BaseModel):
    dataset_name: str

class ChurnInput(BaseModel):
    data: dict

models = {}  # To store trained models

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.get("/datasets/")
def get_datasets():
    # List all available datasets
    data_dir = 'churn_data'
    datasets = list_datasets(data_dir)
    return {"datasets": datasets}

@app.post("/train/")
def train_churn_model(selection: DatasetSelection, model_type: str = "random_forest"):
    global models
    data_dir = 'churn_data'
    file_path = os.path.join(data_dir, selection.dataset_name)
    
    # Load and preprocess the dataset
    df = load_and_preprocess_data(file_path)
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Train the model
    results = train_model(X, y, model_type=model_type)
    
    # Save the model
    models[selection.dataset_name] = results
    
    return {"message": f"Model trained for {selection.dataset_name}", "results": results}

@app.post("/predict/")
def predict_churn(selection: DatasetSelection, input_data: ChurnInput):
    global models
    
    # Ensure model is trained
    if selection.dataset_name not in models:
        return {"error": "Model not trained for this dataset. Please train first."}
    
    model = models[selection.dataset_name]
    
    # Preprocess input data
    data = pd.DataFrame([input_data.data])
    data = load_and_preprocess_data(data)
    
    # Make prediction
    prediction = model.predict(data)
    
    return {"prediction": int(prediction[0])}
