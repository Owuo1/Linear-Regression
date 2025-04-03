import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned and reduced dataset from Task 3
df_reduced = pd.read_csv("task3_output.csv")

# Generate boxplots for outlier detection
numeric_cols = df_reduced.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(20, len(numeric_cols) * 1.5))
for i, col in enumerate(numeric_cols):
    plt.subplot(len(numeric_cols), 1, i + 1)
    sns.boxplot(x=df_reduced[col])
    plt.title(f'Box-Whisker Plot: {col}')
    plt.tight_layout()

plt.show()

# Save the result for use in Task 5
df_reduced.to_csv("task4_output.csv", index=False)