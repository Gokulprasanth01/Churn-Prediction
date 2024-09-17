from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Train a model based on input parameters
def train_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'xgboost':
        model = XGBClassifier()
    else:
        raise ValueError("Model type not supported")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    return evaluate_model(y_test, y_pred, model, X_test)

# Evaluate model with common metrics
def evaluate_model(y_test, y_pred, model, X_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc
    }
