# Re-import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the original dataset
df = pd.read_csv("task1_output.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Fill missing values for numeric columns
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

# 1. Healthcare expenditure and low life expectancy
df['Low Life Expectancy'] = df['Life expectancy'] < 65
low_life_exp = df[df['Life expectancy'] < 65]['Total expenditure']
high_life_exp = df[df['Life expectancy'] >= 65]['Total expenditure']
t_stat1, p_val1 = stats.ttest_ind(low_life_exp, high_life_exp, equal_var=False)

# 2. Correlation between Schooling and Life Expectancy
corr_schooling = df['Schooling'].corr(df['Life expectancy'])

# 3. Correlation between Alcohol and Life Expectancy
corr_alcohol = df['Alcohol'].corr(df['Life expectancy'])

# 4. Correlation between Population and Life Expectancy
corr_population = df['Population'].corr(df['Life expectancy'])

# Return results for discussion
results = {
    "Healthcare Expenditure (T-test p-value)": p_val1,
    "Schooling vs Life Expectancy (Correlation)": corr_schooling,
    "Alcohol vs Life Expectancy (Correlation)": corr_alcohol,
    "Population vs Life Expectancy (Correlation)": corr_population
}

# Return results for discussion
results = {
    "Healthcare Expenditure (T-test p-value)": p_val1,
    "Schooling vs Life Expectancy (Correlation)": corr_schooling,
    "Alcohol vs Life Expectancy (Correlation)": corr_alcohol,
    "Population vs Life Expectancy (Correlation)": corr_population
}

# Display results
print("Task 5 Results:\n")
for key, value in results.items():
    print(f"{key}: {value}")
