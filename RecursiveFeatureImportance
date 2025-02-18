import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Load dataset (replace with actual file path or dataframe)
data = pd.read_csv("your_dataset.csv")

# Define features (X) and target variable (y)
X = data.drop(columns=['waste_percentage'])  # Adjust column name as needed
y = data['waste_percentage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Use RFE for feature selection
selector = RFE(model, n_features_to_select=10)  # Adjust to 10-15 as needed
selector = selector.fit(X_train, y_train)

# Get ranking of features
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': selector.ranking_})
feature_ranking = feature_ranking.sort_values(by='Ranking', ascending=True)

# Select top features
selected_features = feature_ranking[feature_ranking['Ranking'] == 1]['Feature'].tolist()

# Print selected features
print("Top selected features:", selected_features)
print(feature_ranking)

# Train final model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model.fit(X_train_selected, y_train)

# Predict and evaluate (optional, add evaluation metrics as needed)
y_pred = model.predict(X_test_selected)

# Save selected features and rankings to CSV
feature_ranking.to_csv("feature_ranking.csv", index=False)
