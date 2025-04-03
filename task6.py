import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv("task2_output.csv")  # Make sure this file exists in your project folder
df.columns = df.columns.str.strip()

# Fill missing values for numeric columns
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop(columns=['Life expectancy'])
y = df['Life expectancy']

# Split data into training and test sets (75% / 25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

# Define evaluation metrics
def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

train_metrics = evaluate(y_train, y_pred_train)
test_metrics = evaluate(y_test, y_pred_test)
overall_metrics = evaluate(y, y_pred_all)

# Print evaluation results
print("\nModel Performance Summary:")
print("Training Set:", train_metrics)
print("Test Set:", test_metrics)
print("Overall Dataset:", overall_metrics)

# Residual scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_all, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal Line (y = x)')
plt.plot(y, y + 5, 'y--', label='±5 Years')
plt.plot(y, y - 5, 'y--')
plt.plot(y, y + 10, 'r--', label='±10 Years')
plt.plot(y, y - 10, 'r--')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Residual Scatter Plot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual histogram
residuals = y - y_pred_all
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cross-validation (10-fold)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, scoring='r2', cv=kfold)

# Cross-validation boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=cv_scores)
plt.title("10-Fold Cross-Validation R2 Scores")
plt.xlabel("R2 Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print CV stats
print("\nCross-Validation R2 Mean:", round(cv_scores.mean(), 4))
print("Cross-Validation R2 Std Dev:", round(cv_scores.std(), 4))
