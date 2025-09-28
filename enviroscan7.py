# Enviroscan7 Streamlit App
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import requests
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]
OPENWEATHER_KEY = "f931ecc3a4864ae98a35630e7a9f5bc2"

# --- Helper Functions ---
def get_weather(lat, lon, api_key):
    url = "http://api.openweathermap.org/data/2.5/weather"
    resp = requests.get(url, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
    return resp.json() if resp.status_code == 200 else {}

def extract_osm_features(lat, lon, radius=2000):
    features = {}
    point = (lat, lon)
    try:
        roads = ox.features_from_point(point, tags={"highway": True}, dist=radius)
        features["roads_count"] = len(roads)
    except:
        features["roads_count"] = 0
    try:
        industries = ox.features_from_point(point, tags={"landuse": ["industrial", "commercial"]}, dist=radius)
        features["industries_count"] = len(industries)
    except:
        features["industries_count"] = 0
    try:
        farms = ox.features_from_point(point, tags={"landuse": ["farmland", "farm", "agricultural"]}, dist=radius)
        features["farms_count"] = len(farms)
    except:
        features["farms_count"] = 0
    try:
        dumps = ox.features_from_point(point, tags={"landuse": ["landfill", "waste", "dump"]}, dist=radius)
        features["dumps_count"] = len(dumps)
    except:
        features["dumps_count"] = 0
    return features

def build_dataset(city, lat, lon, aq_csv_file, openweather_key):
    try:
        # Debug: Inspect raw file content
        aq_csv_file.seek(0)
        raw_content = aq_csv_file.read().decode('utf-8').splitlines()[:5]
        st.write("First 5 lines of uploaded CSV:", raw_content)
        
        # Reset file pointer for pandas
        aq_csv_file.seek(0)
        df_aq = pd.read_csv(
            aq_csv_file,
            skiprows=0,
            on_bad_lines="skip",
            engine="python",
            encoding='utf-8',
            sep=','
        )
        st.write("Raw CSV columns:", df_aq.columns.tolist())
        st.write("Raw CSV shape:", df_aq.shape)
        df_aq = df_aq.loc[:, ~df_aq.columns.str.contains("^Unnamed")]
        df_aq["source"] = "OpenAQ"
        
        # Debug: Verify stations
        st.write("Unique stations in raw CSV:", df_aq['location_name'].nunique())
        st.write("Stations:", df_aq['location_name'].unique().tolist())
        
    except Exception as e:
        st.error(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), {}
    
    # Ensure latitude and longitude are present
    if 'latitude' not in df_aq.columns or 'longitude' not in df_aq.columns:
        df_aq['latitude'] = lat
        df_aq['longitude'] = lon
    
    # Define expected pollutants
    POLLUTANTS = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    
    # Convert CO from ppb to µg/m³ for consistency
    df_aq.loc[df_aq['parameter'] == 'co', 'value'] *= 1144.6
    
    # Pivot to wide format per location/timestamp
    df_agg = df_aq.groupby(['location_name', 'latitude', 'longitude', 'datetimeUtc', 'parameter'])['value'].mean().reset_index()
    df_wide = df_agg.pivot_table(
        index=['location_name', 'latitude', 'longitude', 'datetimeUtc'],
        columns='parameter',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Initialize missing pollutant columns
    for pollutant in POLLUTANTS:
        if pollutant not in df_wide.columns:
            df_wide[pollutant] = np.nan
    
    # Fill missing values PER STATION
    df_wide = df_wide.sort_values(['location_name', 'datetimeUtc'])
    df_wide[POLLUTANTS] = df_wide.groupby('location_name')[POLLUTANTS].fillna(method='ffill').fillna(method='bfill')
    
    # Add OSM features per unique location
    unique_locations = df_wide[['location_name', 'latitude', 'longitude']].drop_duplicates()
    osm_dict = {}
    for _, row in unique_locations.iterrows():
        osm = extract_osm_features(row['latitude'], row['longitude'], radius=2000)
        key = (row['location_name'], row['latitude'], row['longitude'])
        osm_dict[key] = osm
    
    df_osm = df_wide.apply(lambda row: pd.Series(osm_dict.get((row['location_name'], row['latitude'], row['longitude']),
                                                              {'roads_count': 0, 'industries_count': 0, 'farms_count': 0, 'dumps_count': 0})), axis=1)
    df_wide = pd.concat([df_wide, df_osm], axis=1)
    
    # Add weather data
    weather_data = get_weather(lat, lon, openweather_key)
    weather_features = {
        'temp_c': weather_data.get('main', {}).get('temp'),
        'humidity': weather_data.get('main', {}).get('humidity'),
        'pressure': weather_data.get('main', {}).get('pressure'),
        'wind_speed': weather_data.get('wind', {}).get('speed'),
        'wind_dir': weather_data.get('wind', {}).get('deg'),
        'weather_source': 'OpenWeatherMap'
    }
    for k, v in weather_features.items():
        df_wide[k] = v
    
    # Add derived features
    df_wide['aqi_proxy'] = df_wide.get('pm25', 0) * 0.4 + df_wide.get('pm10', 0) * 0.3 + \
                           df_wide.get('no2', 0) * 0.2 + df_wide.get('co', 0) * 0.1
    df_wide['pollution_per_road'] = df_wide.get('pm25', 0) / (df_wide.get('roads_count', 1) + 1)
    
    # Metadata
    meta = {
        'city': city,
        'latitude': lat,
        'longitude': lon,
        'records': len(df_wide),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'OpenAQ',
        'unique_stations': df_wide['location_name'].nunique()
    }
    return df_wide, meta

def save_datasets(df, filename):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning(f"{filename} is empty. Skipping save.")
        return
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    df.to_csv(f"{filename}.csv", index=False)
    df.to_json(f"{filename}.json", orient="records", indent=2)
    st.success(f"Saved {filename}.csv and {filename}.json")

def consolidate_dataset(df_aq, df_meta, filename):
    if df_aq is None or df_aq.empty:
        st.warning("AQ dataset empty, skipping consolidation.")
        return
    for k, v in df_meta.items():
        df_aq[k] = v
    df_aq.to_csv(f"{filename}.csv", index=False)
    st.success(f"Consolidated dataset saved as {filename}.csv")

def label_source(row):
    pm25 = row.get("pm25", 0)
    roads = row.get("roads_count", 0)
    industries = row.get("industries_count", 0)
    farms = row.get("farms_count", 0)
    
    # More robust labeling logic
    if pd.notna(pm25) and pm25 > 25 and industries > 0:
        return "Industrial"
    elif pd.notna(pm25) and pm25 > 15 and roads > 5:
        return "Traffic"
    elif farms > 0:
        return "Agricultural"
    else:
        return "Mixed/Other"

# --- Streamlit App ---
st.title("Enviroscan Environmental Data Analysis")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    st.info("Processing uploaded file...")
    city = "Delhi"
    lat, lon = 28.7041, 77.1025
    # Build dataset
    df_aq, meta = build_dataset(city, lat, lon, uploaded_file, OPENWEATHER_KEY)
    if not df_aq.empty:
        save_datasets(df_aq, "delhi_aq_data")
        save_datasets(meta, "delhi_meta_data")
        consolidate_dataset(df_aq, meta, "delhi_environmental_data")
        st.success("✅ Dataset processing complete.")
        
        # Use df_aq from build_dataset directly
        if not df_aq.empty:
            df = df_aq
            # Add pollution source labels BEFORE scaling
            required_cols = ['pm25', 'roads_count', 'industries_count', 'farms_count']
            if all(col in df.columns for col in required_cols):
                df['pollution_source'] = df.apply(label_source, axis=1)
            else:
                st.warning("⚠️ Required columns for labeling pollution_source are missing. Skipping label assignment.")
                df['pollution_source'] = 'Unknown'

            # Compute and display median values per station
            st.subheader("Median Values per Station")
            key_columns = ['location_name', 'latitude', 'longitude', 'pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 
                           'roads_count', 'industries_count', 'farms_count', 'dumps_count', 'aqi_proxy', 'pollution_per_road', 
                           'temp_c', 'humidity', 'pressure', 'wind_speed']
            key_columns = [col for col in key_columns if col in df.columns]
            median_df = df[key_columns].groupby(['location_name', 'latitude', 'longitude']).median().reset_index()

            # Add most common pollution source per station
            mode_source = df.groupby('location_name')['pollution_source'].agg(lambda x: x.mode()[0]).reset_index()
            median_df = median_df.merge(mode_source, on='location_name')

            # Display formatted table
            st.dataframe(median_df.style.format({
                'latitude': '{:.6f}', 
                'longitude': '{:.6f}', 
                'pm25': '{:.2f}', 
                'pm10': '{:.2f}', 
                'no2': '{:.2f}', 
                'co': '{:.2f}', 
                'so2': '{:.2f}', 
                'o3': '{:.2f}', 
                'aqi_proxy': '{:.2f}', 
                'pollution_per_road': '{:.2f}', 
                'temp_c': '{:.2f}', 
                'humidity': '{:.2f}', 
                'pressure': '{:.2f}', 
                'wind_speed': '{:.2f}'
            }))
            st.write(f"Total stations: {df['location_name'].nunique()}")
            st.write(f"Total rows in raw data: {len(df)}")

            # Optional: Debug NaNs in medians
            # st.write("NaNs in median table:", median_df.isna().sum())

            # Continue with scaling and saving
            num_cols = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 'roads_count', 'industries_count', 
                        'farms_count', 'dumps_count', 'aqi_proxy', 'pollution_per_road', 
                        'temp_c', 'humidity', 'pressure', 'wind_speed']
            num_cols = [col for col in num_cols if col in df.columns]
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            df.to_csv('cleaned_environmental_data.csv', index=False)
            st.success("💾 Cleaned dataset saved as cleaned_environmental_data.csv")

        # --- Optional: Model Training ---
        if st.button("Train Models and Predict Pollution Source"):
            st.info("Training models...")
            X = df.drop(columns=["pollution_source"])
            y = df["pollution_source"]
            # Ensure consistent samples
            st.write(f"X shape: {X.shape}, y shape: {y.shape}")
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            st.write(f"After cleaning: X shape: {X.shape}, y shape: {y.shape}")
            # Check class distribution
            st.write("Class distribution before resampling:")
            st.write(pd.Series(y).value_counts())
            # Visualize class distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x="pollution_source", data=df, ax=ax, palette="viridis")
            ax.set_title("Pollution Source Distribution")
            ax.set_xlabel("Pollution Source")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            # Keep only numeric features
            X = X.select_dtypes(include=[np.number])
            numeric_columns = X.columns.tolist()
            # Impute and scale
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            # Use simpler models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            }
            # Use cross-validation for small datasets
            if X.shape[0] < 50:
                st.warning("Small dataset detected. Using cross-validation instead of train-test split.")
                for name, model in models.items():
                    scores = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], return_train_score=False)
                    st.write(f"{name} Cross-Validation Results:")
                    st.write(f"Accuracy: {scores['test_accuracy'].mean():.2f} ± {scores['test_accuracy'].std():.2f}")
                    st.write(f"Precision: {scores['test_precision_weighted'].mean():.2f} ± {scores['test_precision_weighted'].std():.2f}")
                    st.write(f"Recall: {scores['test_recall_weighted'].mean():.2f} ± {scores['test_recall_weighted'].std():.2f}")
                    st.write(f"F1: {scores['test_f1_weighted'].mean():.2f} ± {scores['test_f1_weighted'].std():.2f}")
            else:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                from imblearn.over_sampling import SMOTE
                # Balance training data with SMOTE
                if len(y_train.value_counts()) > 1 and min(y_train.value_counts()) > 1:
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    st.write("Class distribution after SMOTE:")
                    st.write(pd.Series(y_train).value_counts())
                else:
                    st.warning("Not enough samples for SMOTE. Proceeding with original training data.")
                
                # Impute and scale
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                performance = {}
                for name, model in models.items():
                    st.write(f"Training {name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    performance[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
                    st.write(f"Test results for {name}:")
                    st.text(classification_report(y_test, y_pred, zero_division=0))
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
                    ax.set_title(f"Confusion Matrix - {name}")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                
                # Save best model
                best_model_name = max(performance, key=lambda k: performance[k]["F1"])
                best_model = models[best_model_name]
                joblib.dump(best_model, "pollution_source_model.pkl")
                st.success(f"💾 Best model saved as pollution_source_model.pkl")
                
                # Save predictions
                X_test_orig = pd.DataFrame(X_test, columns=numeric_columns)
                X_test_orig["actual_source"] = y_test.reset_index(drop=True)
                X_test_orig["predicted_source"] = y_pred
                X_test_orig.to_csv("final_predictions.csv", index=False)
                st.success("💾 Final predictions saved as final_predictions.csv")
