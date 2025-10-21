# Kenya SME Credit Scoring - Main Script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🏢 Kenya SME Credit Scoring Model")
print("=" * 40)

def load_data():
    """Load and return SME data"""
    try:
        df = pd.read_csv('data/kenya_sme_dataset.csv')
        print(f"✅ Data loaded: {df.shape[0]} SMEs, {df.shape[1]} features")
        return df
    except FileNotFoundError:
        print("❌ Data file not found. Please run data generation first.")
        return None

if __name__ == "__main__":
    print("🚀 Starting Kenya SME Credit Scoring System...")
    data = load_data()
    if data is not None:
        print("✅ System ready!")