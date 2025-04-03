import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import combinations

# Load dataset
df = pd.read_csv("task2_output.csv")
df.columns = df.columns.str.strip()

# Fill missing numeric values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop(columns=['Life expectancy'])
y = df['Life expectancy']

# Helper to check how many predictions fall outside ±10 years
def check_within_error_lines(features, X, y, margin=10):
    X_subset = X[list(features)]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_subset)
    errors = np.abs(y_pred - y)
    outside_bounds = np.sum(errors > margin)
    return outside_bounds, list(features)

# Reduce feature search to top N features (ranked by correlation with target)
top_n = 10
correlations = X.corrwith(y).abs().sort_values(ascending=False)
top_features = correlations.head(top_n).index.tolist()

min_outside = float('inf')
best_features = None

# Try combinations of top N features
for r in range(2, len(top_features) + 1):
    for combo in combinations(top_features, r):
        outliers, used = check_within_error_lines(combo, X, y)
        print(f"Checking features: {used} => {outliers} outside ±10 years")
        if outliers == 0:
            min_outside = outliers
            best_features = used
            break
    if best_features:
        break

# Final result output
print("\n Task 7 Results:")
if best_features:
    print("Minimum Features Needed:", len(best_features))
    print("Selected Features:", best_features)
else:
    print("No subset of top features found where all predictions fall within ±10 years.")
