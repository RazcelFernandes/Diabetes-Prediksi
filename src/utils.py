# src/utils.py
import joblib

def save_pickle(obj, file_path):
    """Save object to pickle file"""
    joblib.dump(obj, file_path)

def load_pickle(file_path):
    """Load object from pickle file"""
    return joblib.load(file_path)