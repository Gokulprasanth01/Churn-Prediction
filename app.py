from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = FastAPI()

# Serve static files (like HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

class DatasetSelection(BaseModel):
    dataset_name: str

class ModelSelection(BaseModel):
    dataset_name: str
    model_name: str

class ChurnInput(BaseModel):
    data: dict

# Dictionary to store trained models
models = {}

def create_pipeline(categorical_features, numerical_features):
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create the full pipeline with preprocessing and Random Forest classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

def train_and_evaluate(pipeline, df):
    if 'churn' not in df.columns:
        raise ValueError("'churn' column not found in the dataset")
    
    # Split data into features and target
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc
    }

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.get("/datasets/")
def get_datasets():
    data_dir = 'churn_data'
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    return {"datasets": files}

@app.get("/datasets/{dataset_name}")
def get_dataset_columns(dataset_name: str):
    data_dir = 'churn_data'
    file_path = os.path.join(data_dir, dataset_name)
    
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    df = pd.read_csv(file_path)
    columns = df.columns.tolist()
    
    return {"columns": columns}

@app.get("/models/")
def get_models():
    # List all available models
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    return {"models": files}

@app.post("/train/")
def train_churn_model(selection: DatasetSelection, model_type: str = "random_forest"):
    global models
    data_dir = 'churn_data'
    file_path = os.path.join(data_dir, selection.dataset_name)
    
    # Load and preprocess the dataset
    df = load_and_preprocess_data(file_path)
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Define features for pipeline; you may need to adapt these based on your dataset
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    
    # Create the pipeline
    pipeline = create_pipeline(categorical_features, numerical_features)
    
    # Train and evaluate
    evaluation_results = train_and_evaluate(pipeline, df)
    
    # Save the model
    model_filename = f"models/{selection.dataset_name}.pkl"
    joblib.dump(pipeline, model_filename)
    models[selection.dataset_name] = pipeline
    
    return {"message": f"Model trained for {selection.dataset_name}", "results": evaluation_results}

@app.post("/predict/")
def predict_churn(selection: ModelSelection, input_data: ChurnInput):
    global models
    
    # Ensure model is trained
    model_filename = f"models/{selection.dataset_name}.pkl"
    if not os.path.isfile(model_filename):
        return {"error": "Model not found for this dataset. Please train first."}
    
    model = joblib.load(model_filename)
    
    # Preprocess input data
    data = pd.DataFrame([input_data.data])
    data_dir = 'churn_data'
    file_path = os.path.join(data_dir, selection.dataset_name)
    df = load_and_preprocess_data(file_path)
    categorical_features = [col for col in df.columns if df[col].dtype == 'object' and col != 'churn']
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'churn']
    pipeline = create_pipeline(categorical_features, numerical_features)
    data = pipeline.named_steps['preprocessor'].transform(data)
    
    # Make prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    
    return {"prediction": int(prediction[0]), "probability": probability[0]}
