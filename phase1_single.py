import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -------------------------------
# Helper Function: NRMSE
# -------------------------------
def compute_nrmse(y_true, y_pred):
    """Compute Normalized RMSE = RMSE / (max - min)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    norm_factor = y_true.max() - y_true.min()
    return rmse / norm_factor if norm_factor != 0 else rmse

# -------------------------------
# Load Training Data
# -------------------------------
data_file = 'Grouped_Data.xlsx'
df = pd.read_excel(data_file)

target_col = 'Waste %'
y = df[target_col].astype(np.float32)

# Define all possible features
all_features = [
    'Flute Code Grouped', 'Qty Bucket', 'Component Code Grouped',
    'Machine Group 1', 'Last Operation', 'qty_ordered',
    'number_up_entry_grouped', 'OFFSET?', 'Operation', 'Test Code'
]
X = df[all_features]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data: 80% training, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)

# -------------------------------
# Train XGBoost Model
# -------------------------------
best_params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'random_state': 42
}

model = xgb.XGBRegressor(**best_params)
model.fit(X_train_full, y_train_full)

# -------------------------------
# Dynamic Feature Selection Based on User Input
# -------------------------------
def predict_waste(user_inputs):
    """
    Accepts user_inputs as a dictionary where keys are feature names,
    and values are user-provided inputs (can be numeric or categorical).
    """
    # Convert user input to DataFrame
    df_input = pd.DataFrame([user_inputs])
    
    # One-hot encode based on the training data
    df_input_encoded = pd.get_dummies(df_input, drop_first=True)
    df_input_encoded = df_input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    
    # Predict using the trained model
    prediction = model.predict(df_input_encoded)
    return prediction[0]

# Example Usage
user_selected_features = {
    'Flute Code Grouped': 'A',
    'Qty Bucket': 200,
    'Component Code Grouped': 'X',
    'Machine Group 1': 'M1',
    'qty_ordered': 500
}
predicted_waste = predict_waste(user_selected_features)
print("Predicted Waste %:", predicted_waste)

joblib.dump(model, "trained_model.pkl")