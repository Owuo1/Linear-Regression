import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import combinations

# Load dataset
df = pd.read_csv("task2_output.csv")  # Make sure this file exists
df.columns = df.columns.str.strip()

# Fill missing numeric values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# Convert categorical variables to numeric (including 'Population Size')
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop(columns=['Life expectancy'])
y = df['Life expectancy']

# Helper function: how many predictions fall outside ±10 years
def check_within_error_lines(features, X, y, margin=10):
    X_subset = X[list(features)]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_subset)
    errors = np.abs(y_pred - y)
    outside_bounds = np.sum(errors > margin)
    return outside_bounds, list(features)

# Optimized loop: start with most important features and find the smallest set
feature_names = list(X.columns)
min_outside = float('inf')
best_features = None

for r in range(2, len(feature_names) + 1):
    for combo in combinations(feature_names, r):
        outliers, used = check_within_error_lines(combo, X, y)
        if outliers == 0:  # All predictions within ±10 years
            min_outside = outliers
            best_features = used
            break
    if best_features:
        break  # Stop at first smallest set that satisfies condition

# Output the best feature set
print("\n Task 7 Results:")
if best_features:
    print("Minimum Features Needed:", len(best_features))
    print("Selected Features:", best_features)
else:
    print("❌ No subset found that keeps all predictions within ±10 years.")
