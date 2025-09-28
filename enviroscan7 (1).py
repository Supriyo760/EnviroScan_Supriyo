import streamlit as st
import requests
import pandas as pd
import numpy as np
import osmnx as ox
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import tempfile

# Streamlit page configuration
st.set_page_config(page_title="EnviroScan: Air Quality Analysis", layout="wide")

# Title and description
st.title("🌍 EnviroScan: Air Quality Analysis")
st.markdown("""
This app analyzes air quality data for a given city using OpenAQ, OpenWeatherMap, and OpenStreetMap data.
Upload an air quality CSV file, specify the city and coordinates, and the app will process the data,
train a model to predict pollution sources, and provide downloadable results.
""")

# Weather API Key (replace with your own or use environment variable for security)
openweather_key = "f931ecc3a4864ae98a35630e7a9f5bc2"  # Replace with your key or use os.getenv("OPENWEATHER_API_KEY")

# List of pollutants
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

def get_weather(lat, lon, api_key):
    url = "http://api.openweathermap.org/data/2.5/weather"
    try:
        resp = requests.get(url, params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})
        return resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        st.error(f"Failed to fetch weather data: {e}")
        return {}

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
    st.write(f"📊 Collecting AQ data for {city} from uploaded CSV...")
    
    # Load AQ CSV
    try:
        df_aq = pd.read_csv(aq_csv_file, sep=",", skiprows=2, on_bad_lines="skip", engine="python")
        df_aq = df_aq.loc[:, ~df_aq.columns.str.contains("^Unnamed")]
        df_aq["source"] = "OpenAQ"
    except Exception as e:
        st.error(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), {}

    # Horizontal pivot preview
    try:
        df_preview = df_aq[['location_name', 'parameter', 'value']]
        df_horizontal = df_preview.pivot_table(
            index='location_name',
            columns='parameter',
            values='value',
            aggfunc='first'
        ).reset_index()
        st.write("📊 AQ Dataset Preview (horizontal format):")
        st.dataframe(df_horizontal.head(10))
    except Exception as e:
        st.warning(f"⚠️ Could not pivot for horizontal view: {e}")
        st.dataframe(df_aq.head())

    # Weather features
    weather_data = get_weather(lat, lon, openweather_key)
    weather_features = {
        "temp_c": weather_data.get("main", {}).get("temp", None),
        "humidity": weather_data.get("main", {}).get("humidity", None),
        "pressure": weather_data.get("main", {}).get("pressure", None),
        "wind_speed": weather_data.get("wind", {}).get("speed", None),
        "wind_dir": weather_data.get("wind", {}).get("deg", None),
        "weather_source": "OpenWeatherMap"
    }

    # OSM features
    try:
        st.write(f"Fetching OSM graph for {city} at ({lat}, {lon}) with radius 5000m...")
        G = ox.graph_from_point((lat, lon), dist=5000, network_type="drive")
        road_count = len(G.edges()) if G.edges() else 0
        gdf = ox.geocode_to_gdf(city)
        gdf_utm = gdf.to_crs(epsg=32643)
        area_km2 = gdf_utm.geometry.area.iloc[0] / 1e6
        osm_features = extract_osm_features(lat, lon, radius=2000)
        osm_features["osm_area_km2"] = area_km2
        osm_features["road_count"] = road_count if road_count > 0 else osm_features.get("roads_count", 0)
        osm_features["osm_source"] = "OpenStreetMap/OSMnx"
        st.write(f"Road count: {road_count}, Area: {area_km2} km²")
    except Exception as e:
        st.warning(f"⚠️ OSM data fetch failed: {e}")
        osm_features = extract_osm_features(lat, lon, radius=2000)
        osm_features["osm_area_km2"] = None
        osm_features["road_count"] = osm_features.get("roads_count", 0)
        osm_features["osm_source"] = "OpenStreetMap/OSMnx"

    # Metadata dictionary
    meta = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "records": len(df_aq),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta.update(weather_features)
    meta.update(osm_features)
    st.write("📊 Metadata + Weather + OSM:", meta)

    return df_aq, meta

def save_datasets(df, filename, temp_dir):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning(f"⚠️ {filename} is empty. Skipping save.")
        return None, None
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    csv_path = os.path.join(temp_dir, f"{filename}.csv")
    json_path = os.path.join(temp_dir, f"{filename}.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    st.success(f"💾 Saved {filename}.csv and {filename}.json")
    return csv_path, json_path

def consolidate_dataset(df_aq, df_meta, filename, temp_dir):
    if df_aq is None or df_aq.empty:
        st.warning("⚠️ AQ dataset empty, skipping consolidation.")
        return None
    for k, v in df_meta.items():
        df_aq[k] = v
    csv_path = os.path.join(temp_dir, f"{filename}.csv")
    df_aq.to_csv(csv_path, index=False)
    st.success(f"💾 Consolidated dataset saved as {filename}.csv")
    return csv_path

def clean_and_engineer_features(df):
    st.write("=== Data Cleaning and Feature Engineering ===")
    st.write(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define metadata columns to retain
    meta_cols = ["city", "latitude", "longitude", "timestamp", "roads_count", 
                 "industries_count", "farms_count", "dumps_count", "osm_area_km2", 
                 "road_count", "osm_source", "weather_source", "temp_c", "humidity", 
                 "pressure", "wind_speed", "wind_dir"]

    # Pivot if needed
    if "parameter" in df.columns and "value" in df.columns:
        st.write("Pivoting dataset so pollutants become columns...")
        st.write(f"Before pivot, roads_count: {df['roads_count'].iloc[0] if 'roads_count' in df.columns else 'missing'}")
        pivot_index = ["location_name"] + [col for col in meta_cols if col in df.columns]
        df = df.pivot_table(
            index=pivot_index,
            columns="parameter",
            values="value",
            aggfunc="first"
        ).reset_index()
        df.columns.name = None
        st.write(f"After pivot, roads_count: {df['roads_count'].iloc[0] if 'roads_count' in df.columns else 'missing'}")

    # Ensure OSM features exist
    osm_features = ["roads_count", "industries_count", "farms_count", "dumps_count"]
    for col in osm_features:
        if col not in df.columns:
            df[col] = 0

    # Handle missing values
    df = df.replace({None: np.nan})
    for col in POLLUTANTS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    weather_cols = ["temp_c", "humidity", "pressure", "wind_speed", "wind_dir"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    st.write(f"Removed {before - after} duplicate rows")

    # Drop irrelevant columns
    drop_cols = ["data_source", "openaq_api_version", "openweathermap_api_version", "osmnx_version"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure pollutants exist
    for col in ["pm25", "pm10", "no2", "o3"]:
        if col not in df.columns:
            df[col] = np.nan

    # Create features
    df["aqi_proxy"] = df[["pm25", "pm10", "no2", "o3"]].mean(axis=1)
    df["pollution_per_road"] = df["pm25"] / (df["roads_count"] + 1)

    def categorize_aqi(val):
        if val <= 50: return "Good"
        elif val <= 100: return "Moderate"
        elif val <= 200: return "Unhealthy"
        else: return "Hazardous"

    df["aqi_category"] = df["aqi_proxy"].apply(categorize_aqi)

    # Standardize numeric features
    num_cols = ["pm25", "pm10", "no2", "co", "so2", "o3", "roads_count", 
                "industries_count", "farms_count", "dumps_count", "aqi_proxy", 
                "pollution_per_road"] + weather_cols
    num_cols = [col for col in num_cols if col in df.columns]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    st.write(f"Standardized {len(num_cols)} numerical features")

    # Encode categorical features
    categorical_cols = ["city", "aqi_category"]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        st.write(f"Encoded categorical features: {categorical_cols}")
    else:
        st.warning("⚠️ No categorical columns found to encode")

    return df, scaler

def expand_and_label_dataset(df):
    dfs = []
    for i in range(100):
        temp = df.copy()
        temp["pm25"] += np.random.normal(0, 0.1, size=len(df))
        temp["pm10"] += np.random.normal(0, 0.1, size=len(df))
        temp["roads_count"] += np.random.randint(0, 2, size=len(df))
        temp["industries_count"] += np.random.randint(0, 2, size=len(df))
        temp["farms_count"] += np.random.randint(0, 2, size=len(df))
        dfs.append(temp)
    df_expanded = pd.concat(dfs, ignore_index=True)
    st.write("✅ Expanded dataset shape:", df_expanded.shape)

    def label_source(row):
        if row["pm25"] > 0 and row.get("industries_count", 0) > 0:
            return "Industrial"
        elif row["pm25"] > 0 and row.get("roads_count", 0) > 0:
            return "Traffic"
        elif row.get("farms_count", 0) > 0:
            return "Agricultural"
        else:
            return "Mixed/Other"

    df_expanded["pollution_source"] = df_expanded.apply(label_source, axis=1)
    st.write("✅ Source labels assigned using rule-based simulation")
    st.write(df_expanded["pollution_source"].value_counts())

    return df_expanded

def train_and_evaluate_models(df, temp_dir):
    X = df.drop(columns=["pollution_source"])
    y = df["pollution_source"]
    X = X.select_dtypes(include=[np.number])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    X_test_orig = X_test.copy()

    # Balance dataset
    df_train = pd.concat([X_train, y_train], axis=1)
    majority_class = df_train["pollution_source"].value_counts().idxmax()
    dfs = []
    for label in df_train["pollution_source"].unique():
        subset = df_train[df_train["pollution_source"] == label]
        if label != majority_class:
            subset = resample(subset, replace=True, n_samples=df_train[df_train["pollution_source"] == majority_class].shape[0], random_state=42)
        dfs.append(subset)
    df_train_balanced = pd.concat(dfs)
    X_train = df_train_balanced.drop(columns=["pollution_source"])
    y_train = df_train_balanced["pollution_source"]
    st.write("✅ Applied class balancing")

    # Impute and scale
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
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    performance = {}

    for name, model in models.items():
        st.write(f"🔹 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        performance[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

        st.write(f"Validation Results for {name}:")
        st.text(classification_report(y_val, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)

    best_model_name = max(performance, key=lambda k: performance[k]["F1"])
    best_model = models[best_model_name]
    st.write(f"🏆 Best Model Selected: {best_model_name}")

    # Test set evaluation
    y_test_pred = best_model.predict(X_test)
    st.write("Final Test Performance:")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    # Save model
    model_path = os.path.join(temp_dir, "pollution_source_model.pkl")
    joblib.dump(best_model, model_path)

    # Save predictions
    X_test_orig["actual_source"] = y_test.reset_index(drop=True)
    X_test_orig["predicted_source"] = y_test_pred
    predictions_path = os.path.join(temp_dir, "final_predictions.csv")
    X_test_orig.to_csv(predictions_path, index=False)
    return model_path, predictions_path, best_model_name

# Streamlit UI
with st.form("input_form"):
    st.subheader("Input Parameters")
    city = st.text_input("City Name", value="Delhi")
    lat = st.number_input("Latitude", value=28.7041, format="%.4f")
    lon = st.number_input("Longitude", value=77.1025, format="%.4f")
    aq_csv_file = st.file_uploader("Upload Air Quality CSV (from OpenAQ)", type=["csv"])
    submit_button = st.form_submit_button("Run Analysis")

if submit_button and aq_csv_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temp directory
        aq_csv_path = os.path.join(temp_dir, aq_csv_file.name)
        with open(aq_csv_path, "wb") as f:
            f.write(aq_csv_file.getbuffer())

        # Build dataset
        df_aq, df_meta = build_dataset(city, lat, lon, aq_csv_path, openweather_key)

        # Save datasets
        aq_csv_path, aq_json_path = save_datasets(df_aq, "delhi_aq_data", temp_dir)
        meta_csv_path, meta_json_path = save_datasets(df_meta, "delhi_meta_data", temp_dir)
        consolidated_path = consolidate_dataset(df_aq, df_meta, "delhi_environmental_data", temp_dir)

        # Download buttons
        if aq_csv_path:
            with open(aq_csv_path, "rb") as f:
                st.download_button("Download AQ Data CSV", f, file_name="delhi_aq_data.csv")
        if meta_csv_path:
            with open(meta_csv_path, "rb") as f:
                st.download_button("Download Metadata CSV", f, file_name="delhi_meta_data.csv")
        if consolidated_path:
            with open(consolidated_path, "rb") as f:
                st.download_button("Download Consolidated Data CSV", f, file_name="delhi_environmental_data.csv")

        # Clean and engineer features
        df = pd.read_csv(consolidated_path) if consolidated_path else pd.DataFrame()
        if not df.empty:
            df_clean, scaler = clean_and_engineer_features(df)
            cleaned_path = os.path.join(temp_dir, "cleaned_environmental_data.csv")
            df_clean.to_csv(cleaned_path, index=False)
            st.success("💾 Cleaned dataset saved as cleaned_environmental_data.csv")
            st.write("Cleaned Data Preview:")
            st.dataframe(df_clean.head(10))
            with open(cleaned_path, "rb") as f:
                st.download_button("Download Cleaned Data CSV", f, file_name="cleaned_environmental_data.csv")

            # Expand and label dataset
            df_expanded = expand_and_label_dataset(df_clean)
            labeled_path = os.path.join(temp_dir, "source_labeled_data.csv")
            df_expanded.to_csv(labeled_path, index=False)
            st.success("💾 Source-labeled dataset saved as source_labeled_data.csv")
            with open(labeled_path, "rb") as f:
                st.download_button("Download Source-Labeled Data CSV", f, file_name="source_labeled_data.csv")

            # Train and evaluate models
            model_path, predictions_path, best_model_name = train_and_evaluate_models(df_expanded, temp_dir)
            with open(model_path, "rb") as f:
                st.download_button("Download Trained Model", f, file_name="pollution_source_model.pkl")
            with open(predictions_path, "rb") as f:
                st.download_button("Download Final Predictions CSV", f, file_name="final_predictions.csv")

elif submit_button:
    st.error("Please upload an air quality CSV file to proceed.")
