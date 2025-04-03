import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("task2_output.csv")  # update name if necessary

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Fill missing numeric values (if any remain)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Plot the correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Life Expectancy Predictors")
plt.tight_layout()
plt.show()

# Identify highly correlated features (r > 0.75)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation_cols = [col for col in upper_triangle.columns if any(upper_triangle[col].abs() > 0.75)]

# Drop those columns
df_reduced = df.drop(columns=high_correlation_cols)

# Print the dropped columns
print("\nHighly correlated columns (r > 0.75) that were dropped:")
print(high_correlation_cols)


# ... drop highly correlated features and save reduced dataframe
df_reduced = df.drop(columns=['under-five deaths', 'GDP', 'thinness 5-9 years', 'Schooling'])

df_reduced.to_csv("task3_output.csv", index=False)