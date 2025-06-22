# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# Ensure outputs directory exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Step 1: Load dataset
try:
    data = pd.read_csv('data/advertising.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/advertising.csv' not found. Ensure it's in the 'data/' folder.")
    exit(1)

# Step 2: Exploratory Data Analysis (EDA)
# Inspect dataset
print("\nDataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Summarize statistics
print("\nSummary Statistics:")
print(data.describe())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualize relationships (scatter plots)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(data['TV'], data['Sales'], color='blue', alpha=0.5)
plt.xlabel('TV Spend ($)')
plt.ylabel('Sales (thousands)')
plt.title('TV vs Sales')

plt.subplot(1, 3, 2)
plt.scatter(data['Radio'], data['Sales'], color='green', alpha=0.5)
plt.xlabel('Radio Spend ($)')
plt.ylabel('Sales (thousands)')
plt.title('Radio vs Sales')

plt.subplot(1, 3, 3)
plt.scatter(data['Newspaper'], data['Sales'], color='red', alpha=0.5)
plt.xlabel('Newspaper Spend ($)')
plt.ylabel('Sales (thousands)')
plt.title('Newspaper vs Sales')

plt.tight_layout()
plt.savefig('outputs/scatter_plots.png')
plt.close()  # Close plot to free memory
print("\nScatter plots saved to 'outputs/scatter_plots.png'")

# Correlation matrix
correlation = data.corr()
print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation)), correlation.columns)
plt.title('Correlation Matrix')
plt.savefig('outputs/correlation_matrix.png')
plt.close()
print("Correlation matrix saved to 'outputs/correlation_matrix.png'")

# Export summary statistics to Excel
try:
    data.describe().to_excel('outputs/summary_statistics.xlsx', index=True)
    print("Summary statistics exported to 'outputs/summary_statistics.xlsx'")
except Exception as e:
    print(f"Error exporting summary statistics to Excel: {e}")

# Step 3: Data Preprocessing
# Verify data quality
print("\nDataset Shape:", data.shape)
print("\nMissing Values (Reverified):")
print(data.isnull().sum())

# Select features and target
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']  # Target

print("\nFeatures (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Features Head:")
print(X_scaled_df.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining Set Shape:", X_train.shape, y_train.shape)
print("Test Set Shape:", X_test.shape, y_test.shape)

# Export preprocessed data to Excel
try:
    train_data = pd.DataFrame(X_train, columns=X.columns)
    train_data['Sales'] = y_train.values
    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['Sales'] = y_test.values
    train_data.to_excel('outputs/train_data.xlsx', index=False)
    test_data.to_excel('outputs/test_data.xlsx', index=False)
    print("Preprocessed data exported to 'outputs/train_data.xlsx' and 'outputs/test_data.xlsx'")
except Exception as e:
    print(f"Error exporting preprocessed data to Excel: {e}")

# Step 4: Linear Regression Modeling
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance:")
print(f"R² Score: {r2:.3f}")
print(f"Mean Squared Error: {mse:.3f}")

# Feature coefficients
coeff_df = pd.DataFrame(model.coef_, index=['TV', 'Radio', 'Newspaper'], columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coeff_df)

# Visualize actual vs predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales (thousands)')
plt.ylabel('Predicted Sales (thousands)')
plt.title('Actual vs Predicted Sales')
plt.tight_layout()
plt.savefig('outputs/actual_vs_predicted.png')
plt.close()
print("Actual vs predicted plot saved to 'outputs/actual_vs_predicted.png'")

# Export predictions to Excel
try:
    results = pd.DataFrame({'Actual Sales': y_test.values, 'Predicted Sales': y_pred})
    results.to_excel('outputs/test_predictions.xlsx', index=False)
    print("Test predictions exported to 'outputs/test_predictions.xlsx'")
except Exception as e:
    print(f"Error exporting test predictions to Excel: {e}")

# Save model and scaler for Step 5
try:
    joblib.dump(model, 'outputs/linear_regression_model.pkl')
    joblib.dump(scaler, 'outputs/standard_scaler.pkl')
    print("Model saved to 'outputs/linear_regression_model.pkl'")
    print("Scaler saved to 'outputs/standard_scaler.pkl'")
except Exception as e:
    print(f"Error saving model/scaler: {e}")