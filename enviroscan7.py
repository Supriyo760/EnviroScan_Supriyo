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

def extract_osm_features(lat, lon, radius=100):
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
        df_aq = df_aq.loc[:, ~df_aq.columns.str.contains("^Unnamed")]
        df_aq["source"] = "OpenAQ"
    except Exception as e:
        print(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), pd.DataFrame()

        # ✅ Create metadata as a DataFrame
    metadata = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "records": len(df_aq),
        "source": "OpenAQ"
    }
    df_meta = pd.DataFrame([metadata])

    return df_aq, df_meta

    # Weather
    weather_data = get_weather(lat, lon, openweather_key)
    weather_features = {
        "temp_c": weather_data.get("main", {}).get("temp"),
        "humidity": weather_data.get("main", {}).get("humidity"),
        "pressure": weather_data.get("main", {}).get("pressure"),
        "wind_speed": weather_data.get("wind", {}).get("speed"),
        "wind_dir": weather_data.get("wind", {}).get("deg"),
        "weather_source": "OpenWeatherMap"
    }

    # OSM Features
    osm_features = extract_osm_features(lat, lon, radius=2000)

    # Metadata
    meta = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "records": len(df_aq),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    meta.update(weather_features)
    meta.update(osm_features)

    return df_aq, meta

def save_datasets(df, filename):
    if df is None or df.empty:
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
    
    if pm25 > 0 and industries > 0:
        return "Industrial"
    elif pm25 > 0 and roads > 0:
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
    df_aq, df_meta = build_dataset(city, lat, lon, uploaded_file, OPENWEATHER_KEY)

    if not df_aq.empty:
        save_datasets(df_aq, "delhi_aq_data")
        save_datasets(df_meta, "delhi_meta_data")
        consolidate_dataset(df_aq, df_meta, "delhi_environmental_data")

        st.success("✅ Dataset processing complete.")

        # --- Data Cleaning ---
        df = pd.read_csv("delhi_environmental_data.csv")

        # Pivot pollutants
        # Pivot pollutants safely
        # --- Determine pivot index dynamically ---
        pivot_index = [col for col in ["location_name", "city", "latitude", "longitude", "timestamp"] if col in df.columns]
        
        if "parameter" in df.columns and "value" in df.columns:
            df = df.pivot_table(
                index=pivot_index,
                columns="parameter",
                values="value",
                aggfunc="first"  # or 'mean' if multiple readings per timestamp
            ).reset_index()
            df.columns.name = None
            st.success("✅ Pivoted data with all stations and pollutants")
        else:
            st.warning("⚠️ 'parameter' or 'value' columns missing, cannot pivot")


        # Fill missing pollutants
        # Keep all main pollutants
        pollutant_cols = [c for c in ["pm25","pm10","no2","o3","co","so2"] if c in df.columns]
        
        for col in pollutant_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill missing weather
        weather_cols = ["temp_c","humidity","pressure","wind_speed","wind_dir"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Ensure OSM features
        for col in ["roads_count","industries_count","farms_count","dumps_count"]:
            if col not in df.columns:
                df[col] = 0

        # Create features
        pollutant_cols = [c for c in ["pm25","pm10","no2","o3"] if c in df.columns]
        
        if pollutant_cols:
            df["aqi_proxy"] = df[pollutant_cols].mean(axis=1)
        else:
            df["aqi_proxy"] = np.nan
            st.warning("⚠️ No pollutant columns found, aqi_proxy set to NaN")
        
        # Pollution per road
        if "pm25" in df.columns and "roads_count" in df.columns:
            df["pollution_per_road"] = df["pm25"] / (df["roads_count"] + 1)
        else:
            df["pollution_per_road"] = np.nan
            st.warning("⚠️ pm25 or roads_count missing, skipping pollution_per_road")
        
        # AQI category
        df["aqi_category"] = df["aqi_proxy"].apply(
            lambda x: (
                "Good" if pd.notna(x) and x <= 50 else
                "Moderate" if pd.notna(x) and x <= 100 else
                "Unhealthy" if pd.notna(x) and x <= 200 else
                "Hazardous"
            )
        )

        # Standardize numeric columns
        num_cols = ["pm25","pm10","no2","co","so2","o3","roads_count","industries_count","farms_count","dumps_count","aqi_proxy","pollution_per_road"] + weather_cols
        num_cols = [col for col in num_cols if col in df.columns]
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Encode categorical
        categorical_cols = ["city","aqi_category"]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Assign pollution sources
        required_cols = ["pm25", "roads_count", "industries_count", "farms_count"]
        if any(col in df.columns for col in required_cols):
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

            # Keep only numeric features
            X = X.select_dtypes(include=[np.number])

            # Split
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

            # Balance training
            df_train = pd.concat([X_train, y_train], axis=1)
            majority_class = df_train["pollution_source"].value_counts().idxmax()
            dfs = []
            for label in df_train["pollution_source"].unique():
                subset = df_train[df_train["pollution_source"]==label]
                if label != majority_class:
                    subset = resample(subset, replace=True, n_samples=df_train[df_train["pollution_source"]==majority_class].shape[0], random_state=42)
                dfs.append(subset)
            df_train_balanced = pd.concat(dfs)
            X_train = df_train_balanced.drop(columns=["pollution_source"])
            y_train = df_train_balanced["pollution_source"]

            # Impute + scale
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            X_test = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "Neural Network": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
            }

            performance = {}
            for name, model in models.items():
                st.write(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

                performance[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
                st.write(f"Validation results for {name}:")
                st.text(classification_report(y_val, y_pred, zero_division=0))

                # Confusion matrix plot
                cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
                fig, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
                ax.set_title(f"Confusion Matrix - {name}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            best_model_name = max(performance, key=lambda k: performance[k]["F1"])
            best_model = models[best_model_name]
            st.success(f"🏆 Best model selected: {best_model_name}")

            # Evaluate on test set
            y_test_pred = best_model.predict(X_test)
            st.subheader("Final Test Performance")
            st.text(classification_report(y_test, y_test_pred, zero_division=0))

            # Save model + predictions
            joblib.dump(best_model, "pollution_source_model.pkl")
            st.success("💾 Best model saved as pollution_source_model.pkl")

            X_test_orig = pd.DataFrame(X_test, columns=X.select_dtypes(include=[np.number]).columns)
            X_test_orig["actual_source"] = y_test.reset_index(drop=True)
            X_test_orig["predicted_source"] = y_test_pred
            X_test_orig.to_csv("final_predictions.csv", index=False)
            st.success("💾 Final predictions saved as final_predictions.csv")
