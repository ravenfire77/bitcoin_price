import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BitcoinPredictionAnalysis:
    def __init__(self):
        self.btc_data = None
        self.predictions = None
        self.model = None
        
    def load_or_download_data(self, file_path=None):
        """Load Bitcoin data from file or download from Yahoo Finance"""
        if file_path:
            try:
                print(f"Loading Bitcoin data from {file