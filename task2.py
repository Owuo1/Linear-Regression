import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (ensure CSV is in the same folder as this script)
df = pd.read_csv("task1_output.csv")  # Rename if needed

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Display missing value counts before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

# Impute numeric missing values with the mean of each column
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Drop any remaining rows with missing values (e.g., if non-numeric)
df.dropna(inplace=True)

# Display missing value counts after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sort_values(ascending=False))

# Set target variable
target = 'Life expectancy'

# Get numeric predictor variables excluding the target
predictors = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]

# Plot scatter plots: predictor vs life expectancy
num_cols = 4
num_rows = (len(predictors) // num_cols) + int(len(predictors) % num_cols > 0)

plt.figure(figsize=(20, num_rows * 4))

for i, col in enumerate(predictors):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.scatterplot(x=df[col], y=df[target])
    plt.xlabel(col)
    plt.ylabel('Life Expectancy')
    plt.title(f'{col} vs Life Expectancy')

plt.tight_layout()
plt.show()

# Save Task 2 output for next step
df.to_csv("task2_output.csv", index=False)
