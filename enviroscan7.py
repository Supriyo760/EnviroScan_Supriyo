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
from sklearn.model_selection import cross_validate, KFold
from imblearn.over_sampling import SMOTE
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
        # Try to read with multiple delimiters and skip comment lines
        delimiters = [',', '\t', ';']
        df_aq = None
        for sep in delimiters:
            try:
                aq_csv_file.seek(0)
                df_aq = pd.read_csv(
                    aq_csv_file,
                    comment="#",     # <-- ignore junk headers like "# CSV-File created..."
                    on_bad_lines="skip",
                    engine="python",
                    sep=sep
                )
                if df_aq.shape[1] > 1:  # valid parse
                    st.write(f"Successfully loaded CSV with delimiter '{sep}'")
                    break
            except Exception as e:
                st.write(f"Failed with delimiter '{sep}': {e}")
        if df_aq is None or df_aq.empty:
            st.error("⚠️ Failed to load usable AQ CSV.")
            return pd.DataFrame(), {}

        st.write("Raw CSV columns:", df_aq.columns.tolist())
        st.write("Raw CSV shape:", df_aq.shape)

        # Case 1: Already wide format (pollutants are columns)
        if any(col.lower() in POLLUTANTS for col in df_aq.columns):
            st.info("Detected wide-format CSV with pollutant columns.")
            if "location_name" not in df_aq.columns:
                df_aq["location_name"] = "UnknownStation"
            if "datetimeUtc" not in df_aq.columns:
                df_aq["datetimeUtc"] = datetime.now(timezone.utc)
            df_wide = df_aq.copy()

        # Case 2: Long format (parameter + value)
        elif "parameter" in df_aq.columns and "value" in df_aq.columns:
            st.info("Detected long-format CSV with parameter/value pairs.")
            if "location_name" not in df_aq.columns:
                df_aq["location_name"] = "UnknownStation"
            if "datetimeUtc" not in df_aq.columns:
                df_aq["datetimeUtc"] = datetime.now(timezone.utc)

            df_agg = df_aq.groupby(['location_name', 'datetimeUtc', 'parameter'])['value'].mean().reset_index()
            df_wide = df_agg.pivot_table(
                index=['location_name', 'datetimeUtc'],
                columns='parameter',
                values='value',
                aggfunc='mean'
            ).reset_index()
        else:
            st.error("⚠️ CSV format not recognized. Expected pollutant columns or (parameter,value).")
            return pd.DataFrame(), {}

        # Ensure pollutant columns exist
        for pollutant in POLLUTANTS:
            if pollutant not in df_wide.columns:
                df_wide[pollutant] = np.nan

        # Add static lat/lon if not present
        if 'latitude' not in df_wide.columns:
            df_wide['latitude'] = lat
        if 'longitude' not in df_wide.columns:
            df_wide['longitude'] = lon

        # OSM features
        unique_locations = df_wide[['location_name', 'latitude', 'longitude']].drop_duplicates()
        osm_dict = {}
        for _, row in unique_locations.iterrows():
            osm = extract_osm_features(row['latitude'], row['longitude'])
            osm_dict[(row['location_name'], row['latitude'], row['longitude'])] = osm
        df_osm = df_wide.apply(
            lambda row: pd.Series(osm_dict.get((row['location_name'], row['latitude'], row['longitude']),
                                               {'roads_count': 0, 'industries_count': 0, 'farms_count': 0, 'dumps_count': 0})), axis=1)
        df_wide = pd.concat([df_wide, df_osm], axis=1)

        # Weather data
        weather_data = get_weather(lat, lon, openweather_key)
        df_wide['temp_c'] = weather_data.get('main', {}).get('temp')
        df_wide['humidity'] = weather_data.get('main', {}).get('humidity')

        # Derived features
        df_wide['aqi_proxy'] = df_wide[POLLUTANTS].mean(axis=1)
        df_wide['pollution_per_road'] = df_wide['pm25'] / (df_wide.get('roads_count', 1) + 1)

        meta = {
            'city': city,
            'latitude': lat,
            'longitude': lon,
            'records': len(df_wide),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unique_stations': df_wide['location_name'].nunique()
        }
        return df_wide, meta

    except Exception as e:
        st.error(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), {}

# -------------------------
# (Rest of your script stays SAME as you pasted — 
# save_datasets, consolidate_dataset, label_source, Streamlit app code, etc.)
# -------------------------
