import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import sys

# -------------------------------
# Helper Function: NRMSE
# -------------------------------
def compute_nrmse(y_true, y_pred):
    """Compute Normalized RMSE = RMSE / (max - min)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    norm_factor = y_true.max() - y_true.min()
    return rmse / norm_factor if norm_factor != 0 else rmse

# -------------------------------
# 1. Data Loading & Preprocessing (Main Training Data)
# -------------------------------
data_file = 'Grouped_Data.xlsx'
df = pd.read_excel(data_file)

target_col = 'Waste %'
y = df[target_col].astype(np.float32)

selected_features = [
    'Flute Code Grouped',
    'Qty Bucket',
    'Component Code Grouped',
    'Machine Group 1',
    'Last Operation',
    'qty_ordered',
    'number_up_entry_grouped',
    'OFFSET?',
    'Operation',
    'Test Code'
]
X = df[selected_features]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 2. Split data: 80% training, 20% test
# -------------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)

# -------------------------------
# 3. XGBoost Hyperparameters
# -------------------------------
best_params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# -------------------------------
# 4. Prepare for Iterations
# -------------------------------
n_iterations = 100

# Containers for NRMSE
new_train_nrmse_list = []
new_test_nrmse_list = []
original_test_nrmse_list = []

# Containers for MSE
new_train_mse_list = []
new_test_mse_list = []
original_test_mse_list = []

# List to store each trained model
trained_models = []

# -------------------------------
# 5. Main Loop (100 iterations)
# -------------------------------
for i in range(n_iterations):
    # Split the original training data (80%) into new training (80%) and new testing (20%)
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=i
    )

    # Train an XGBoost model on the new training split
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_new_train, y_new_train)

    # Store this trained model
    trained_models.append(model)

    # Predictions on new training and new testing splits
    pred_new_train = model.predict(X_new_train)
    pred_new_test = model.predict(X_new_test)

    # Compute and store MSE for new training and new testing
    mse_new_train = mean_squared_error(y_new_train, pred_new_train)
    mse_new_test = mean_squared_error(y_new_test, pred_new_test)
    new_train_mse_list.append(mse_new_train)
    new_test_mse_list.append(mse_new_test)

    # Compute and store NRMSE for new training and new testing
    nrmse_new_train = compute_nrmse(y_new_train, pred_new_train)
    nrmse_new_test = compute_nrmse(y_new_test, pred_new_test)
    new_train_nrmse_list.append(nrmse_new_train)
    new_test_nrmse_list.append(nrmse_new_test)

    # Predict on the original 20% test set
    pred_original_test = model.predict(X_test)
    mse_original_test = mean_squared_error(y_test, pred_original_test)
    original_test_mse_list.append(mse_original_test)

    nrmse_original_test = compute_nrmse(y_test, pred_original_test)
    original_test_nrmse_list.append(nrmse_original_test)

# -------------------------------
# 6. Summaries & Outputs (Training Performance)
# -------------------------------
# A) NRMSE Averages
avg_new_train_nrmse = np.mean(new_train_nrmse_list)
avg_new_test_nrmse = np.mean(new_test_nrmse_list)
avg_original_test_nrmse = np.mean(original_test_nrmse_list)

print("Average New Training NRMSE (100 iterations):", avg_new_train_nrmse)
print("Average New Testing NRMSE (100 iterations):", avg_new_test_nrmse)
print("Average Original Testing NRMSE (100 iterations):", avg_original_test_nrmse)

# B) MSE Averages
avg_new_train_mse = np.mean(new_train_mse_list)
avg_new_test_mse = np.mean(new_test_mse_list)
avg_original_test_mse = np.mean(original_test_mse_list)

print("\nAverage New Training MSE (100 iterations):", avg_new_train_mse)
print("Average New Testing MSE (100 iterations):", avg_new_test_mse)
print("Average Original Testing MSE (100 iterations):", avg_original_test_mse)

# ------------------------------------------------------------------
# 7. Predicting on a New Jobs File (Ignoring PURCHASED BOARD/OFFSET)
# ------------------------------------------------------------------
new_path = "JobsToPredict.xlsx"
if not os.path.exists(new_path):
        sys.exit("Error: Something went wrong.")
else:
    new_jobs_file = "JobsToPredict.xlsx"

df_jobs = pd.read_excel(new_jobs_file)

# (A) Filter out the rows where 'Machine Group 1' == 'PURCHASED BOARD/OFFSET'
df_jobs = df_jobs[~df_jobs['Machine Group 1'].str.strip().eq('PURCHASED BOARD/OFFSET')]

# (B) Keep columns needed for prediction (matching your training features)
feature_cols = [
    'Flute Code',  # must match exactly what you had in training
    'Qty Bucket',
    'Component Code',
    'Machine Group 1',
    'Last Operation',
    'qty_ordered',
    'number_up_entry_1',
    'OFFSET?',
    'Operation',
    'Test Code'
]

X_jobs = df_jobs[feature_cols].copy()

# (C) One-hot encode new jobs data & align columns
X_jobs_encoded = pd.get_dummies(X_jobs, drop_first=True)
X_jobs_encoded = X_jobs_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# (D) Predict using all 100 trained models
all_preds = []
for model in trained_models:
    preds = model.predict(X_jobs_encoded)
    all_preds.append(preds)

# Convert list to NumPy array, shape: (100, num_rows_in_new_jobs)
all_preds = np.array(all_preds)
# Transpose to shape: (num_rows_in_new_jobs, 100)
all_preds = all_preds.T

# Compute mean & std across the 100 predictions for each row
df_jobs['pred_mean'] = all_preds.mean(axis=1)
df_jobs['pred_std'] = all_preds.std(axis=1)

# (E) Group by job_number and Machine Group 1 and also carry over qty_ordered for each job.
#     Here we use the first qty_ordered encountered per group.
group_cols = ['job_number', 'Machine Group 1']
grouped = df_jobs.groupby(group_cols).agg({
    'pred_mean': 'mean',
    'pred_std': 'mean',
    'qty_ordered': 'first'
}).reset_index()

# Display grouped results
print("\nPredictions for each job-machine combination (mean & std over 100 models) with qty_ordered:")
print(grouped)

# -------------------------------
# 8. Exporting Predictions to Excel
# -------------------------------
output_file_grouped = 'Predicted_Jobs_Grouped.xlsx'
grouped.to_excel(output_file_grouped, index=False)
print("Grouped predictions (with qty_ordered) saved to", output_file_grouped)
