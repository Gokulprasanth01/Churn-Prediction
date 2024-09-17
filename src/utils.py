import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# List all datasets
def list_datasets(data_dir):
    return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    # Encoding categorical columns
    df = encode_categorical_columns(df)
    
    # Standardize numeric columns
    df = standardize_numeric_columns(df)
    
    return df

# Encode categorical columns
def encode_categorical_columns(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        if df[column].nunique() <= 2:
            df[column] = label_encoder.fit_transform(df[column])
        else:
            df = pd.get_dummies(df, columns=[column], drop_first=True)
    return df

# Standardize numeric columns
def standardize_numeric_columns(df):
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Perform basic EDA
def perform_eda(df):
    print(df.head())
    
    # Churn distribution
    sns.countplot(data=df, x='churn')
    plt.title('Churn Distribution')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
