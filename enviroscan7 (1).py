# Enviroscan7 Streamlit App
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import requests
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import smtplib
from email.mime.text import MIMEText
from folium.plugins import HeatMap

# --- Constants ---
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]
OPENWEATHER_KEY = "f931ecc3a4864ae98a35630e7a9f5bc2"
THRESHOLDS = {"pm25": 50, "pm10": 100, "no2": 80, "co": 10000, "so2": 75, "o3": 70}
EMAIL_CONFIG = {
    "sender": "supriyochowdhury760@gmail.com",
    "password": "ryrl xtng ocix nvbn",
    "receiver": "danger49491358@gmail.com",
    "server": "smtp.gmail.com",
    "port": 587
}

# --- Helper Functions ---
def get_weather(lat, lon, api_key):
    """Fetch weather data from OpenWeatherMap API."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {}

def extract_osm_features(lat, lon, radius=2000):
    """Extract OSM features (roads, industries, farms, dumps) around a point."""
    features = {"roads_count": 0, "industries_count": 0, "farms_count": 0, "dumps_count": 0}
    point = (lat, lon)
    try:
        roads = ox.features_from_point(point, tags={"highway": True}, dist=radius)
        features["roads_count"] = len(roads)
    except Exception:
        pass
    try:
        industries = ox.features_from_point(point, tags={"landuse": ["industrial", "commercial"]}, dist=radius)
        features["industries_count"] = len(industries)
    except Exception:
        pass
    try:
        farms = ox.features_from_point(point, tags={"landuse": ["farmland", "farm", "agricultural"]}, dist=radius)
        features["farms_count"] = len(farms)
    except Exception:
        pass
    try:
        dumps = ox.features_from_point(point, tags={"landuse": ["landfill", "waste", "dump"]}, dist=radius)
        features["dumps_count"] = len(dumps)
    except Exception:
        pass
    return features

def build_dataset(city, lat, lon, aq_csv_file, openweather_key):
    """Build dataset from AQ CSV, OSM features, and weather data."""
    try:
        df_aq = pd.read_csv(aq_csv_file, skiprows=2, on_bad_lines="skip", engine="python")
        df_aq = df_aq.loc[:, ~df_aq.columns.str.contains("^Unnamed")]
        df_aq["source"] = "OpenAQ"
    except Exception as e:
        st.error(f"⚠️ Failed to load AQ CSV: {e}")
        return pd.DataFrame(), {}

    if 'latitude' not in df_aq.columns or 'longitude' not in df_aq.columns:
        df_aq['latitude'] = lat
        df_aq['longitude'] = lon

    df_aq.loc[df_aq['parameter'] == 'co', 'value'] *= 1144.6
    df_agg = df_aq.groupby(['location_name', 'latitude', 'longitude', 'datetimeUtc', 'parameter'])['value'].mean().reset_index()
    df_wide = df_agg.pivot_table(index=['location_name', 'latitude', 'longitude', 'datetimeUtc'],
                                 columns='parameter', values='value', aggfunc='mean', fill_value=np.nan).reset_index()

    for pollutant in POLLUTANTS:
        if pollutant not in df_wide.columns:
            df_wide[pollutant] = np.nan

    df_wide = df_wide.sort_values(['location_name', 'datetimeUtc'])
    df_wide[POLLUTANTS] = df_wide.groupby('location_name')[POLLUTANTS].fillna(method='ffill').fillna(method='bfill')

    unique_locations = df_wide[['location_name', 'latitude', 'longitude']].drop_duplicates()
    osm_dict = {(row['location_name'], row['latitude'], row['longitude']): extract_osm_features(row['latitude'], row['longitude'])
                for _, row in unique_locations.iterrows()}
    df_osm = df_wide.apply(lambda row: pd.Series(osm_dict.get((row['location_name'], row['latitude'], row['longitude']),
                                                             {'roads_count': 0, 'industries_count': 0, 'farms_count': 0, 'dumps_count': 0})), axis=1)
    df_wide = pd.concat([df_wide, df_osm], axis=1)

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

    df_wide['aqi_proxy'] = (df_wide.get('pm25', 0) * 0.4 + df_wide.get('pm10', 0) * 0.3 +
                            df_wide.get('no2', 0) * 0.2 + df_wide.get('co', 0) * 0.1)
    df_wide['pollution_per_road'] = df_wide.get('pm25', 0) / (df_wide.get('roads_count', 1) + 1)

    meta = {
        'city': city, 'latitude': lat, 'longitude': lon, 'records': len(df_wide),
        'timestamp': datetime.now(timezone.utc).isoformat(), 'source': 'OpenAQ',
        'unique_stations': df_wide['location_name'].nunique()
    }
    return df_wide, meta

def save_datasets(df, filename):
    """Save dataset to CSV and JSON files."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.warning(f"{filename} is empty. Skipping save.")
        return
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    df.to_csv(f"{filename}.csv", index=False)
    df.to_json(f"{filename}.json", orient="records", indent=2)
    st.success(f"Saved {filename}.csv and {filename}.json")

def consolidate_dataset(df_aq, df_meta, filename):
    """Consolidate AQ data with metadata."""
    if df_aq is None or df_aq.empty:
        st.warning("AQ dataset empty, skipping consolidation.")
        return
    for k, v in df_meta.items():
        df_aq[k] = v
    df_aq.to_csv(f"{filename}.csv", index=False)
    st.success(f"Consolidated dataset saved as {filename}.csv")

def label_source(row):
    """Label pollution source based on PM2.5, roads, industries, and farms."""
    pm25 = row.get("pm25", 0)
    roads = row.get("roads_count", 0)
    industries = row.get("industries_count", 0)
    farms = row.get("farms_count", 0)
    if pd.notna(pm25) and pm25 > 25 and industries > 0:
        return "Industrial"
    elif pd.notna(pm25) and pm25 > 15 and roads > 5:
        return "Traffic"
    elif farms > 0:
        return "Agricultural"
    return "Mixed/Other"

def generate_pdf_report(df, filename, time_range):
    """Generate a PDF report of pollution data."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Enviroscan Pollution Report: {time_range}")
    c.drawString(100, 730, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    c.drawString(100, 700, f"Stations: {df['location_name'].nunique()}")
    y = 680
    for station in df['location_name'].unique():
        station_data = df[df['location_name'] == station][POLLUTANTS].mean().round(2)
        c.drawString(100, y, f"Station: {station}")
        for pollutant, value in station_data.items():
            c.drawString(120, y - 20, f"{pollutant}: {value}")
            y -= 20
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

def send_email_alert(pollutant, value, station, threshold):
    """Send email alert for exceeding pollution thresholds."""
    msg = MIMEText(f"Alert: {pollutant.upper()} level ({value:.2f}) exceeds threshold ({threshold}) at {station} on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.")
    msg['Subject'] = f"Enviroscan Pollution Alert: {pollutant.upper()}"
    msg['From'] = EMAIL_CONFIG["sender"]
    msg['To'] = EMAIL_CONFIG["receiver"]
    
    try:
        with smtplib.SMTP(EMAIL_CONFIG["server"], EMAIL_CONFIG["port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
            server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["receiver"], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")
        return False

def create_folium_map(df, start_date=None, end_date=None, source_filter=None, location_filter=None, show_heatmap=True, heatmap_field='aqi_proxy'):
    """Create a Folium map with heatmap and markers."""
    required_cols = ['latitude', 'longitude', 'location_name', 'datetimeUtc', 'aqi_proxy', 'pollution_source']
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning("DataFrame is empty or missing required columns for map visualization.")
        return None

    df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])
    if start_date:
        df = df[df['datetimeUtc'].dt.date >= start_date]
    if end_date:
        df = df[df['datetimeUtc'].dt.date <= end_date]
    if source_filter and source_filter != "All":
        df = df[df['pollution_source'] == source_filter]
    if location_filter:
        df = df[(df['latitude'] >= location_filter.get('min_lat', -90)) &
                (df['latitude'] <= location_filter.get('max_lat', 90)) &
                (df['longitude'] >= location_filter.get('min_lon', -180)) &
                (df['longitude'] <= location_filter.get('max_lon', 180))]

    if df.empty:
        st.warning("No data available after applying filters.")
        return None

    st.write("Stations before aggregation:", df['location_name'].unique().tolist())
    aggregated_df = df.groupby('location_name').agg({
        'latitude': 'mean', 'longitude': 'mean', 'aqi_proxy': 'mean', 'pollution_source': 'first'
    }).reset_index()
    st.write("Aggregated stations:", aggregated_df['location_name'].tolist())

    center_lat = aggregated_df['latitude'].mean()
    center_lon = aggregated_df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    if show_heatmap and heatmap_field in aggregated_df.columns:
        st.write(f"Generating heatmap for {heatmap_field} with {len(aggregated_df)} points")
        aggregated_df[heatmap_field] = pd.to_numeric(aggregated_df[heatmap_field], errors='coerce')
        aggregated_df[['latitude', 'longitude']] = aggregated_df[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')
        aggregated_df = aggregated_df.dropna(subset=[heatmap_field, 'latitude', 'longitude'])
        if aggregated_df.empty:
            st.warning("No valid numeric data for heatmap after dropping NaN values.")
            return m

        valid_values = aggregated_df[heatmap_field].dropna()
        if len(valid_values) == 0 or valid_values.max() == valid_values.min():
            st.warning("Invalid range for heatmap normalization.")
            normalized_val = [1.0] * len(aggregated_df)
        else:
            normalized_val = ((aggregated_df[heatmap_field] - valid_values.min()) / (valid_values.max() - valid_values.min())).fillna(1.0).tolist()

        heat_data = [[row.latitude, row.longitude, norm_val] for row, norm_val in zip(aggregated_df.itertuples(), normalized_val)
                     if pd.notna(norm_val) and pd.notna(row.latitude) and pd.notna(row.longitude)]
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, gradient={0.0: 'blue', 0.5: 'lime', 1.0: 'red'}).add_to(m)
        else:
            st.warning("No valid data for heatmap.")

    for _, row in aggregated_df.iterrows():
        if pd.isna(row['aqi_proxy']):
            continue
        severity_color = 'green' if row['aqi_proxy'] <= 50 else 'yellow' if row['aqi_proxy'] <= 100 else 'orange' if row['aqi_proxy'] <= 200 else 'red'
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['location_name']}<br>AQI Proxy: {row['aqi_proxy']:.2f}<br>Source: {row['pollution_source']}",
            icon=folium.Icon(color=severity_color, icon='cloud', prefix='fa')
        ).add_to(m)

    return m

# --- Streamlit App ---
st.title("Enviroscan Environmental Data Dashboard")

# --- Input Section ---
with st.sidebar:
    st.subheader("📍 Input Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.text_input("City", value="Delhi", key="city_input")
    with col2:
        lat = st.number_input("Latitude", value=28.7041, format="%.4f", key="lat_input")
    with col3:
        lon = st.number_input("Longitude", value=77.1025, format="%.4f", key="lon_input")
    
    st.subheader("📅 Time Range")
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Start Date", value=datetime(2025, 9, 1), key="start_date")
    with col5:
        end_date = st.date_input("End Date", value=datetime(2025, 9, 15), key="end_date")
    time_range = f"{start_date} to {end_date}"

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")

# --- Main Content ---
if uploaded_file:
    st.info("Processing uploaded file...")
    df_aq, meta = build_dataset(city, lat, lon, uploaded_file, OPENWEATHER_KEY)
    if not df_aq.empty:
        st.write(f"Dataset Summary: {meta['records']} records, {meta['unique_stations']} unique stations")
        st.write("Stations:", df_aq['location_name'].unique().tolist())
        save_datasets(df_aq, "delhi_aq_data")
        save_datasets(meta, "delhi_meta_data")
        consolidate_dataset(df_aq, meta, "delhi_environmental_data")
        st.success("✅ Dataset processing complete.")

        # --- Data Cleaning ---
        df = pd.read_csv("delhi_environmental_data.csv")
        st.write("Columns in DataFrame:", df.columns.tolist())
        df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'], errors='coerce')
        df_filtered = df[(df['datetimeUtc'].dt.date >= start_date) & (df['datetimeUtc'].dt.date <= end_date)]
        if df_filtered.empty:
            st.warning("No data available for the selected time range. Showing all data.")
            df_filtered = df.copy()

        pollutant_cols = [col for col in POLLUTANTS if col in df_filtered.columns]
        for col in pollutant_cols:
            df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
        weather_cols = ["temp_c", "humidity", "pressure", "wind_speed", "wind_dir"]
        for col in weather_cols:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mean())
        for col in ["roads_count", "industries_count", "farms_count", "dumps_count"]:
            df_filtered[col] = df_filtered.get(col, 0)

        if pollutant_cols:
            df_filtered["aqi_proxy"] = df_filtered[pollutant_cols].mean(axis=1, skipna=True)
        else:
            df_filtered["aqi_proxy"] = 0
            st.warning("⚠️ No pollutant columns found, aqi_proxy set to 0 for visualization")
        if "pm25" in df_filtered.columns and "roads_count" in df_filtered.columns:
            df_filtered["pollution_per_road"] = df_filtered["pm25"] / (df_filtered["roads_count"] + 1)
        else:
            df_filtered["pollution_per_road"] = np.nan
            st.warning("⚠️ pm25 or roads_count missing, skipping pollution_per_road")

        df_filtered["aqi_category"] = df_filtered["aqi_proxy"].apply(
            lambda x: "Good" if pd.notna(x) and x <= 50 else
                      "Moderate" if pd.notna(x) and x <= 100 else
                      "Unhealthy" if pd.notna(x) and x <= 200 else "Hazardous")
        if all(col in df_filtered.columns for col in ["pm25", "roads_count", "industries_count", "farms_count"]):
            df_filtered["pollution_source"] = df_filtered.apply(label_source, axis=1)
        else:
            st.warning("⚠️ Required columns for labeling pollution_source are missing.")
            df_filtered["pollution_source"] = "Unknown"

        num_cols = [col for col in ["pm25", "pm10", "no2", "co", "so2", "o3", "roads_count",
                                    "industries_count", "farms_count", "dumps_count", "aqi_proxy",
                                    "pollution_per_road"] + weather_cols if col in df_filtered.columns]
        scaler = StandardScaler()
        df_filtered[num_cols] = scaler.fit_transform(df_filtered[num_cols])
        categorical_cols = [col for col in ["city", "aqi_category"] if col in df_filtered.columns]
        if categorical_cols:
            df_filtered = pd.get_dummies(df_filtered, columns=categorical_cols, drop_first=True)
        df_filtered.to_csv("cleaned_environmental_data.csv", index=False)
        st.success("💾 Cleaned dataset saved as cleaned_environmental_data.csv")

        st.write("Filtered stations count:", df_filtered.groupby('location_name').size())

        # --- Preview Section ---
        st.write("Unique stations:", df_filtered['location_name'].nunique())
        st.write("Stations:", df_filtered['location_name'].unique().tolist())
        st.subheader("📊 AQ Dataset Preview")
        if not df_filtered.empty:
            preview_df = df_filtered.groupby('location_name').head(2).reset_index(drop=True)
            st.dataframe(preview_df.style.format({"aqi_proxy": "{:.2f}"}))
            st.write(f"Displaying up to 2 rows per station. Total stations: {df_filtered['location_name'].nunique()}")
        else:
            st.warning("No data available for preview.")

        # --- Alerts ---
        st.subheader("🚨 Real-Time Alerts")
        for pollutant, threshold in THRESHOLDS.items():
            if pollutant in df_filtered.columns:
                exceedances = df_filtered[df_filtered[pollutant] > threshold]
                if not exceedances.empty:
                    st.error(f"Alert: {pollutant.upper()} exceeds threshold ({threshold}) at {len(exceedances)} records!")
                    st.dataframe(exceedances[['location_name', 'datetimeUtc', pollutant]])
                    for _, row in exceedances.groupby(['location_name', pollutant]).first().reset_index().iterrows():
                        if send_email_alert(pollutant, row[pollutant], row['location_name'], threshold):
                            st.success(f"Email alert sent for {pollutant.upper()} at {row['location_name']}")

        # --- Visual Components ---
        st.subheader("📈 Pollutant Trends")
        for pollutant in POLLUTANTS:
            if pollutant in df_filtered.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                for station in df_filtered['location_name'].unique():
                    station_data = df_filtered[df_filtered['location_name'] == station]
                    ax.plot(station_data['datetimeUtc'], station_data[pollutant], label=station, alpha=0.7)
                ax.set_title(f"{pollutant.upper()} Trend")
                ax.set_xlabel("Time")
                ax.set_ylabel(pollutant.upper())
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        st.subheader("🥧 Pollution Source Distribution")
        fig, ax = plt.subplots(figsize=(6, 6))
        source_counts = df_filtered['pollution_source'].value_counts()
        ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(source_counts)))
        ax.set_title("Predicted Pollution Source Distribution")
        st.pyplot(fig)

        # --- Analysis Options ---
        st.subheader("⚙️ Analysis Options")
        analysis_option = st.radio("Select Analysis:", ("Generate Heatmaps Only", "Train Models"))

        if analysis_option == "Generate Heatmaps Only":
            st.info("Generating heatmaps...")
            with st.spinner("Rendering map..."):
                m = create_folium_map(df_filtered, start_date, end_date)
                if m:
                    st_folium(m, width=800, height=500)
                else:
                    st.warning("Map rendering failed due to data issues.")

        elif analysis_option == "Train Models":
            if st.button("Start Training", key="train_model_button_1"):
                st.info("Training models...")
                df_model = df_filtered.dropna(subset=["pollution_source"]).reset_index(drop=True)
                X = df_model.drop(columns=["pollution_source", "location_name", "datetimeUtc"])
                y = df_model["pollution_source"]
                X = X.select_dtypes(include=[np.number])
                numeric_columns = X.columns.tolist()
                imputer = SimpleImputer(strategy="median")
                X = imputer.fit_transform(X)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
                }

                if X.shape[0] < 50:
                    for name, model in models.items():
                        scores = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                               scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
                        st.write(f"{name} Cross-Validation Results:")
                        st.write({k.replace('test_', ''): f"{v.mean():.2f} ± {v.std():.2f}" for k, v in scores.items()})
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if len(y_train.value_counts()) > 1 and min(y_train.value_counts()) > 1:
                        smote = SMOTE(random_state=42)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.write("Class distribution after SMOTE:", pd.Series(y_train).value_counts())
                    X_train = imputer.fit_transform(X_train)
                    X_test = imputer.transform(X_test)
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    performance = {}
                    for name, model in models.items():
                        with st.spinner(f"Training {name}..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                            }
                            st.write(f"✅ {name} Results: {metrics}")
                            st.text(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                            cm = confusion_matrix(y_test, y_pred, labels=y.unique())
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique(), ax=ax)
                            ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix - {name}")
                            st.pyplot(fig)

                            model_filename = f"{name.replace(' ', '_').lower()}_model.pkl"
                            joblib.dump(model, model_filename)
                            st.success(f"Model saved as {model_filename}")
                            performance[name] = metrics

                    st.subheader("📊 Model Performance Comparison")
                    perf_df = pd.DataFrame(performance).T
                    st.dataframe(perf_df.style.format("{:.2f}"))
                    best_model = perf_df['f1'].idxmax()
                    st.success(f"🏆 Best model based on F1-score: {best_model}")

                    X_test_orig = pd.DataFrame(X_test, columns=numeric_columns)
                    X_test_orig["actual_source"] = y_test.reset_index(drop=True)
                    X_test_orig["predicted_source"] = y_pred
                    X_test_orig["confidence"] = [max(proba) for proba in y_proba] if y_proba is not None else np.nan
                    X_test_orig.to_csv("final_predictions.csv", index=False)
                    st.success("💾 Final predictions saved as final_predictions.csv")

        # --- Map Filters ---
        st.subheader("🗺️ Map Filters")
        col6, col7, col8 = st.columns(3)
        with col6:
            source_filter = st.selectbox("Pollution Source", ["All"] + ["Industrial", "Traffic", "Agricultural", "Mixed/Other"], key="source_filter")
        with col7:
            show_heatmap = st.checkbox("Show Heatmap", value=True, key="show_heatmap")
            if show_heatmap:
                heatmap_fields = [col for col in ['aqi_proxy'] + POLLUTANTS if col in df_filtered.columns]
                heatmap_field = st.selectbox("Heatmap Field", heatmap_fields, key="heatmap_field")
        with col8:
            min_lat = st.number_input("Min Latitude", value=28.0, format="%.4f", key="min_lat")
            max_lat = st.number_input("Max Latitude", value=29.0, format="%.4f", key="max_lat")
            min_lon = st.number_input("Min Longitude", value=76.0, format="%.4f", key="min_lon")
            max_lon = st.number_input("Max Longitude", value=78.0, format="%.4f", key="max_lon")
        location_filter = {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon}

        st.subheader("🗺️ Pollution Map")
        with st.spinner("Rendering map..."):
            map_obj = create_folium_map(df_filtered, start_date, end_date, source_filter, location_filter, show_heatmap, heatmap_field)
            if map_obj:
                st_folium(map_obj, width=700, height=500, key="pollution_map")
            else:
                st.warning("Unable to render map due to data issues.")

        # --- Download Options ---
        st.subheader("📥 Download Reports")
        latest_date = df_filtered['datetimeUtc'].dt.date.max()
        daily_df = df_filtered[df_filtered['datetimeUtc'].dt.date == latest_date]
        st.download_button("Download Daily Report (CSV)", data=daily_df.to_csv(index=False),
                           file_name=f"daily_report_{latest_date}.csv", mime="text/csv")
        weekly_df = df_filtered[df_filtered['datetimeUtc'].dt.date >= latest_date - timedelta(days=7)]
        st.download_button("Download Weekly Report (CSV)", data=weekly_df.to_csv(index=False),
                           file_name=f"weekly_report_{latest_date}.csv", mime="text/csv")
        pdf_buffer = generate_pdf_report(df_filtered, "pollution_report", time_range)
        st.download_button("Download Summary Report (PDF)", data=pdf_buffer,
                           file_name="pollution_report.pdf", mime="application/pdf")
