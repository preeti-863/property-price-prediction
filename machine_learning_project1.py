# =====================================
# California Housing - Regression Model
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load and Inspect Data
# ---------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df


# ---------------------------
# 2. Preprocessing Function
# ---------------------------

def preprocess_data(df):
    # Fill missing values
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

    # Add location clusters using KMeans
    coords = df[['latitude', 'longitude']]
    df['location_cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(coords)

    # Encode ocean_proximity
    proximity_map = {
        'INLAND': 1,
        '<1H OCEAN': 2,
        'NEAR OCEAN': 3,
        'NEAR BAY': 4,
        'ISLAND': 5
    }
    df['proximity_score'] = df['ocean_proximity'].map(proximity_map)
    
    # Filter low proximity scores (keep closer to ocean)
    df = df[df['proximity_score'] > 2]

    # Adjusted target value
    df['adjusted_house_value'] = df['median_house_value'] / df['proximity_score']

    # Bin and encode housing age
    bins = [0, 5, 10, 20, 100]
    labels = ['0-5', '6-10', '11-20', '20+']
    df['housing_median_age'] = pd.cut(df['housing_median_age'], bins=bins, labels=labels)
    
    le = LabelEncoder()
    df['housing_median_age'] = le.fit_transform(df['housing_median_age'].astype(str))

    # One-hot encode ocean proximity
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    return df, le, proximity_map




# ---------------------------
# 3. Train Models
# ---------------------------

def train_models(df):
    X = df.drop(['median_house_value', 'adjusted_house_value'], axis=1)
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple Linear Regression
    simple_model = LinearRegression()
    simple_model.fit(X_train[['median_income']], y_train)
    y_pred_simple = simple_model.predict(X_test[['median_income']])
    print("\nSimple Linear Regression:")
    print(f"R² Score: {r2_score(y_test, y_pred_simple):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_simple)) :,.2f}")

    # Multiple Linear Regression
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred_multi = multi_model.predict(X_test)
    print("\nMultiple Linear Regression:")
    print(f"R² Score: {r2_score(y_test, y_pred_multi):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_multi)):,.2f}")

    # Show coefficients
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': multi_model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)
    print("\nTop Feature Coefficients:")
    print(coef_df.head(10))

    # Return multi_model, feature columns, and test data + predictions for EDA
    return multi_model, X.columns, y_test, y_pred_multi


# ---------------------------
# 4. EDA Visualization
# ---------------------------

def run_eda(df, y_test=None, y_pred=None, model=None, feature_cols=None):

    print("\nSample data (first 5 rows):")
    print(df.head())

    # Show descriptive statistics
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))
    
    # your EDA plotting code here

   
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='median_income', y='median_house_value', data=df)
    plt.title('Median Income vs Median House Value')
    plt.show()

    # 📊 1. Distribution of Target Variable
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='median_house_value', bins=30, kde=True, color='skyblue')  # x not y
    plt.title("Distribution of Median House Value")
    plt.xlabel("Median House Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


    # 📊 2. Correlation Heatmap (Numeric Features Only)
    plt.figure(figsize=(10, 7))
    numeric_cols = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 📈 3. Actual vs Predicted Plot
    if y_test is not None and y_pred is not None:
         plt.figure(figsize=(8, 6))
         sns.scatterplot(x=range(len(y_test)), y=y_test, label="Actual", color='blue', alpha=0.6)
         sns.scatterplot(x=range(len(y_pred)), y=y_pred, label="Predicted", color='orange', alpha=0.6)
         plt.title("Actual vs Predicted Median House Value")
         plt.xlabel("Sample Index")
         plt.ylabel("House Value")
         plt.legend()
         plt.tight_layout()
         plt.show()

    # 📉 4. Residual Plot (Prediction Errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True, color='orange')
    plt.title("Distribution of Residuals (Prediction Errors)")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 🔍 5. Feature Importance (Coefficients)
    coef_df = pd.DataFrame({
       'Feature': feature_cols ,
       'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df.head(10), x='Coefficient', y='Feature', palette='viridis')
    plt.title("Top 10 Most Influential Features")
    plt.tight_layout()
    plt.show()
 
# ---------------------------
# 5. Predict from User Input
# ---------------------------

def predict_from_input(model, feature_cols, proximity_map, le):
    print("\n📥 Enter property details to predict house value:")
    
    try:
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))
        housing_age = int(input("Housing Median Age (years): "))
        total_rooms = int(input("Total Rooms: "))
        total_bedrooms = int(input("Total Bedrooms: "))
        population = int(input("Population in Area: "))
        households = int(input("Number of Households: "))
        median_income = float(input("Median Income (e.g., 3.5): "))
        ocean_proximity = input("Ocean Proximity (INLAND, <1H OCEAN, NEAR OCEAN, NEAR BAY, ISLAND): ").strip().upper()

        if ocean_proximity not in proximity_map or proximity_map[ocean_proximity] <= 2:
            print("❌ Invalid or excluded proximity category. Please enter one of: NEAR OCEAN, NEAR BAY, ISLAND")
            return
         
        proximity_score = proximity_map[ocean_proximity]

        if median_income < 0 or total_rooms < 1 or population < 1:
            print("❌ Please enter realistic values.")
            return

        # Encode housing age bin
        if housing_age <= 5:
            age_label = '0-5'
        elif housing_age <= 10:
            age_label = '6-10'
        elif housing_age <= 20:
            age_label = '11-20'
        else:
            age_label = '20+'
        housing_median_age = le.transform([age_label])[0]

        # Create input DataFrame
        input_data = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'proximity_score': proximity_score,
            'location_cluster': 0,  # Assume default cluster
            'ocean_proximity_<1H OCEAN': 0,
            'ocean_proximity_INLAND': 0,
            'ocean_proximity_ISLAND': 0,
            'ocean_proximity_NEAR BAY': 0,
            'ocean_proximity_NEAR OCEAN': 0
        }

        # Set appropriate ocean proximity one-hot
        onehot_key = f"ocean_proximity_{ocean_proximity}"
        if onehot_key in input_data:
            input_data[onehot_key] = 1

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        predicted_value = model.predict(input_df)[0]
        print(f"\n💰 Predicted Median House Value: ${predicted_value:,.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")


# ---------------------------
# 6. Main Execution
# ---------------------------

def main():
    df = load_data('data_file.csv')
    df, le, proximity_map = preprocess_data(df)
    
    model, feature_cols, y_test, y_pred = train_models(df)

    # Pass test and prediction data for plots
    run_eda(df, y_test=y_test, y_pred=y_pred, model=model, feature_cols=feature_cols)
    
    predict_from_input(model, feature_cols, proximity_map, le)



if __name__ == "__main__":
    main()
