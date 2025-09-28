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

def extract_osm_features(lat, lon, radius=2000):  # Increased radius for better feature capture
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
        df_aq = pd.read_csv(
            aq_csv_file,
            skiprows=2,
            on_bad_lines="skip",
            engine="python"
        )
        st.write("Raw AQ data shape:", df_aq.shape)
        #st.write("Raw AQ data columns:", df_aq.columns.tolist())
        df_aq = df_aq.loc[:, ~df_aq.columns.str.contains("^Unnamed")]
        df_aq["source"] = "OpenAQ"
    except Exception as e:
        st.error(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), {}

    # Ensure latitude and longitude are present
    if 'latitude' not in df_aq.columns or 'longitude' not in df_aq.columns:
        df_aq['latitude'] = lat
        df_aq['longitude'] = lon

    # Pivot to wide format for pollutants per location and timestamp (keep time dimension)
    df_agg = df_aq.groupby(['location_name', 'latitude', 'longitude', 'datetimeUtc', 'parameter'])['value'].last().reset_index()  # Use 'last' or 'mean' if multiple per time; adjust as needed
    df_wide = df_agg.pivot_table(
        index=['location_name', 'latitude', 'longitude', 'datetimeUtc'],
        columns='parameter',
        values='value'
    ).reset_index()

    # Initialize pollutant columns if missing
    for pollutant in POLLUTANTS:
        if pollutant not in df_wide.columns:
            df_wide[pollutant] = np.nan

    # Get unique locations and their OSM features
    unique_locations = df_wide[['location_name', 'latitude', 'longitude']].drop_duplicates()
    osm_dict = {}
    for _, row in unique_locations.iterrows():
        osm_lat, osm_lon = row['latitude'], row['longitude']
        osm = extract_osm_features(osm_lat, osm_lon, radius=2000)
        key = (row['location_name'], osm_lat, osm_lon)  # Unique key for location
        osm_dict[key] = osm
    
    # Add OSM features to df_wide
    def add_osm(row):
        key = (row['location_name'], row['latitude'], row['longitude'])
        return pd.Series(osm_dict.get(key, {}))
    
    df_osm = df_wide.apply(add_osm, axis=1)
    df_wide = pd.concat([df_wide, df_osm], axis=1)

    # Fetch weather data (city-level)
    weather_data = get_weather(lat, lon, openweather_key)
    #st.write("Weather Data:", weather_data)
    weather_features = {
        "temp_c": weather_data.get("main", {}).get("temp"),
        "humidity": weather_data.get("main", {}).get("humidity"),
        "pressure": weather_data.get("main", {}).get("pressure"),
        "wind_speed": weather_data.get("wind", {}).get("speed"),
        "wind_dir": weather_data.get("wind", {}).get("deg"),
        "weather_source": "OpenWeatherMap"
    }
    for k, v in weather_features.items():
        df_wide[k] = v

    # Metadata
    meta = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "records": len(df_wide),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "OpenAQ"
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
    if pd.notna(pm25) and pm25 > 25 and industries > 0:  # Adjusted threshold for pm25
        return "Industrial"
    elif pd.notna(pm25) and pm25 > 15 and roads > 5:  # Adjusted threshold for pm25 and roads
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

        # --- Data Cleaning ---
        df = pd.read_csv("delhi_environmental_data.csv")

        # --- Preview ---
        st.subheader("📊 AQ Dataset Preview")
        st.dataframe(df.head(10))

        # --- Fill missing pollutants ---
        pollutant_cols = [c for c in POLLUTANTS if c in df.columns]
        for col in pollutant_cols:
            df[col] = df[col].fillna(df[col].median())

        # --- Fill missing weather ---
        weather_cols = ["temp_c", "humidity", "pressure", "wind_speed", "wind_dir"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # --- Ensure OSM features exist ---
        for col in ["roads_count", "industries_count", "farms_count", "dumps_count"]:
            if col not in df.columns:
                df[col] = 0

        # --- Create features ---
        if pollutant_cols:
            df["aqi_proxy"] = df[pollutant_cols].mean(axis=1)
        else:
            df["aqi_proxy"] = np.nan
            st.warning("⚠️ No pollutant columns found, aqi_proxy set to NaN")

        if "pm25" in df.columns and "roads_count" in df.columns:
            df["pollution_per_road"] = df["pm25"] / (df["roads_count"] + 1)
        else:
            df["pollution_per_road"] = np.nan
            st.warning("⚠️ pm25 or roads_count missing, skipping pollution_per_road")

        df["aqi_category"] = df["aqi_proxy"].apply(
            lambda x: (
                "Good" if pd.notna(x) and x <= 50 else
                "Moderate" if pd.notna(x) and x <= 100 else
                "Unhealthy" if pd.notna(x) and x <= 200 else
                "Hazardous"
            )
        )

        # --- Standardize numeric columns ---
        num_cols = ["pm25", "pm10", "no2", "co", "so2", "o3", "roads_count", "industries_count", "farms_count", "dumps_count", "aqi_proxy", "pollution_per_road"] + weather_cols
        num_cols = [col for col in num_cols if col in df.columns]
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # --- Encode categorical ---
        categorical_cols = ["city", "aqi_category"]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # --- Assign pollution sources ---
        required_cols = ["pm25", "roads_count", "industries_count", "farms_count"]
        if all(col in df.columns for col in required_cols):
            df["pollution_source"] = df.apply(label_source, axis=1)
        else:
            st.warning("⚠️ Required columns for labeling pollution_source are missing. Skipping label assignment.")
            df["pollution_source"] = "Unknown"

        # Save cleaned dataset
        df.to_csv("cleaned_environmental_data.csv", index=False)
        st.success("💾 Cleaned dataset saved as cleaned_environmental_data.csv")

        # Display preview
        st.subheader("Preview of Cleaned Dataset")
        st.dataframe(df.head(10))

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
            if X.shape[0] < 50:  # Arbitrary threshold for small datasets
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
                X_test_orig = pd.DataFrame(X_test, columns=X.select_dtypes(include=[np.number]).columns)
                X_test_orig["actual_source"] = y_test.reset_index(drop=True)
                X_test_orig["predicted_source"] = y_pred
                X_test_orig.to_csv("final_predictions.csv", index=False)
                st.success("💾 Final predictions saved as final_predictions.csv")
