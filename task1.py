import pandas as pd

# Load your CSV (make sure it's in the same folder as this script)
df = pd.read_csv("Life_Expectancy_Data.csv")

# Preview the first few rows
print(df.head())

# Clean column names if needed (removes leading/trailing spaces)
df.columns = df.columns.str.strip()

# Population Size category
def categorize_population(pop):
    if pd.isna(pop):
        return 'Unknown'
    elif pop < 30000:
        return 'Small'
    elif 30000 <= pop < 100000:
        return 'Medium'
    else:
        return 'Large'

df['Population Size'] = df['Population'].apply(categorize_population)

# Lifestyle: Alcohol + BMI
df['Lifestyle'] = df['Alcohol'] + df['BMI']

# Economy: Population × GDP
df['Economy'] = df['Population'] * df['GDP']

# Death Ratio: Adult Mortality ÷ Infant Deaths (add 1 to avoid division by zero)
df['Death Ratio'] = df['Adult Mortality'] / (df['infant deaths'] + 1)

# Preview updated data
print(df[['Country', 'Population Size', 'Lifestyle', 'Economy', 'Death Ratio']].head())

# At the end of task1.py
df.to_csv("task1_output.csv", index=False)

