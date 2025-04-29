import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Task 1: Load and Explore the Dataset ---
try:
    air_quality_df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
except FileNotFoundError:
    print("Error: 'AirQualityUCI.csv' not found in the current directory. Please download it from the UCI repository.")
    exit()

print("First 5 rows of the air quality dataset:")
print(air_quality_df.head())
print("\n" + "="*50 + "\n")

print("Information about the air quality dataset:")
air_quality_df.info()
print("\n" + "="*50 + "\n")

print("Number of missing values in each column:")
print(air_quality_df.isnull().sum())
print("\n" + "="*50 + "\n")

# Convert 'Date' and 'Time' columns to a single datetime index
air_quality_df['DateTime'] = pd.to_datetime(air_quality_df['Date'] + ' ' + air_quality_df['Time'], format='%d/%m/%Y %H.%M.%S')
air_quality_df.set_index('DateTime', inplace=True)

# Drop the original 'Date' and 'Time' columns
air_quality_df.drop(columns=['Date', 'Time'], inplace=True)

print("\nDataset info after combining Date and Time:")
air_quality_df.info()
print("\n" + "="*50 + "\n")

# Handle missing values: Replace -200.0 with NaN, then fill NaN with the mean
air_quality_df.replace(to_replace=-200.0, value=np.nan, inplace=True)
air_quality_df.fillna(air_quality_df.mean(numeric_only=True), inplace=True)

print("Number of missing values after handling:")
print(air_quality_df.isnull().sum())
print("\n" + "="*50 + "\n")

# --- Task 2: Basic Data Analysis ---

print("Descriptive statistics of numerical columns:")
print(air_quality_df.describe())
print("\n" + "="*50 + "\n")

# Example: Average CO levels by month
monthly_avg_co = air_quality_df['CO(GT)'].groupby(air_quality_df.index.month).mean()
print("Average monthly CO(GT) levels (mg/m³):")
print(monthly_avg_co)
print("\n" + "="*50 + "\n")

# Example: Average temperature by day of the week
daily_avg_temp = air_quality_df['T'].groupby(air_quality_df.index.day_name()).mean()
print("Average Temperature (°C) by day of the week:")
print(daily_avg_temp)
print("\n" + "="*50 + "\n")

# --- Task 3: Data Visualization ---

# 1. Line chart of CO levels over time
plt.figure(figsize=(12, 6))
plt.plot(air_quality_df['CO(GT)'], label='CO(GT) (mg/m³)')
plt.title('Carbon Monoxide Concentration Over Time')
plt.xlabel('Date')
plt.ylabel('CO(GT) (mg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n(Line chart of CO over time)\n")

# 2. Bar chart of average monthly CO levels
plt.figure(figsize=(8, 6))
sns.barplot(x=monthly_avg_co.index, y=monthly_avg_co.values, palette='viridis')
plt.title('Average Monthly Carbon Monoxide Concentration')
plt.xlabel('Month')
plt.ylabel('Average CO(GT) (mg/m³)')
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()
print("\n(Bar chart of average monthly CO)\n")

# 3. Histogram of Temperature
plt.figure(figsize=(8, 6))
plt.hist(air_quality_df['T'].dropna(), bins=20, edgecolor='black', color='skyblue')
plt.title('Distribution of Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()
print("\n(Histogram of Temperature)\n")

# 4. Scatter plot of Temperature vs. Humidity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='T', y='RH', data=air_quality_df)
plt.title('Relationship between Temperature and Relative Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Humidity (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n(Scatter plot of Temperature vs. Humidity)\n")

# --- Findings and Observations ---
print("\n--- Initial Findings and Observations ---")
print("Based on the initial analysis of the Air Quality dataset:")
print("- The dataset contains hourly readings of various air pollutants and meteorological conditions.")
print("- Missing values are represented by -200.0 and have been replaced with NaN and then imputed using the mean of each column.")
print(f"- The average monthly Carbon Monoxide (CO) levels show variations throughout the year.")
print(f"- The average temperature appears to differ across the days of the week.")
print("- The line chart shows the trend of CO concentration over the time period of the dataset.")
print("- The scatter plot provides a visual indication of the relationship (or lack thereof) between temperature and relative humidity.")
# Add more observations based on the specific patterns you see in the data and visualizations.

print("\n--- End of Analysis ---")