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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
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
