#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 00:29:13 2024

@author: abdur-rahmanibn-bilalwaajid
"""
import pandas as pd
import joblib

# Global variables
model = None
scaler = None
imputer = None
lags_ = None  # Placeholder for lags data

def predict(test: pd.DataFrame, lags: pd.DataFrame | None) -> pd.DataFrame:
    """Make a prediction."""
    global model, scaler, imputer, lags_
    
    # Update global lags data
    if lags is not None:
        lags_ = lags
    
    # Load model, scaler, and imputer during the first call
    if model is None:
        print("Loading model and preprocessors...")
        model = joblib.load('models/lightgbm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        imputer = joblib.load('models/imputer.pkl')
    
    # Preprocessing
    features = test.drop(columns=["row_id"])  # Adjust based on your feature set
    features = imputer.transform(features)
    features = scaler.transform(features)
    
    # Prediction
    test['responder_6'] = model.predict(features)
    
    # Return predictions
    predictions = test[['row_id', 'responder_6']]
    assert len(predictions) == len(test)
    return predictions

