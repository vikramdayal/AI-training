import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Create Example Data
# ---------------------------------------------------------

np.random.seed(42)

num_rows = 1000

df = pd.DataFrame({
    "square_feet": np.random.randint(500, 4000, num_rows),
    "bedrooms": np.random.randint(1, 6, num_rows),
    "age": np.random.randint(0, 50, num_rows),
    "distance_to_city": np.random.uniform(1, 40, num_rows)
})

# Create target variable with some noise
df["price"] = (
    df["square_feet"] * 250
    + df["bedrooms"] * 15000
    - df["age"] * 1200
    - df["distance_to_city"] * 3000
    + np.random.normal(0, 25000, num_rows)
)

print("Sample Data:")
print(df.head())

# ---------------------------------------------------------
# 2. Define Features and Target
# ---------------------------------------------------------

X = df[[
    "square_feet",
    "bedrooms",
    "age",
    "distance_to_city"
]]

y = df["price"]

# ---------------------------------------------------------
# 3. Split Into Training and Test Sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------
# 4. Train Regression Models
# ---------------------------------------------------------

linear_model = LinearRegression()

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Make Predictions
# ---------------------------------------------------------

linear_preds = linear_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# ---------------------------------------------------------
# 6. Evaluate Models
# ---------------------------------------------------------

def evaluate_model(name, y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

evaluate_model("Linear Regression", y_test, linear_preds)
evaluate_model("Random Forest Regression", y_test, rf_preds)

# ---------------------------------------------------------
# 7. Compare Predictions
# ---------------------------------------------------------

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "LinearPrediction": linear_preds,
    "RandomForestPrediction": rf_preds
})

print("\nPrediction Samples:")
print(results_df.head(10))

# ---------------------------------------------------------
# 8. Feature Importance (Random Forest)
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(importance_df)
