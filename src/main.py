# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Set working directory
os.chdir('/Users/abdur-rahmanibn-bilalwaajid/jane_street_project')
print(f"Working directory set to: {os.getcwd()}")

# Define expected features
EXPECTED_FEATURES = [f"feature_{i:02}" for i in range(93)]

# Initialize preprocessing tools and model
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
model = None  # Model initialized after determining feature size

# Main function
if __name__ == "__main__":
    try:
        # Step 1: Load the partitioned dataset using Dask
        print("Loading dataset with Dask...")
        dask_data = dd.read_parquet('data/train.parquet', engine='pyarrow')
        print(f"Dataset loaded successfully! Shape: {dask_data.shape}")
        
        # Step 2: Process the data in chunks
        print("Processing data in chunks...")
        for i, chunk in enumerate(dask_data.to_delayed()):  # Iterate over Dask chunks
            chunk = chunk.compute()  # Convert to Pandas DataFrame
            print(f"Processing chunk {i + 1} with shape: {chunk.shape}")
            
            try:
                # Ensure all expected features are present (fill missing features)
                for col in EXPECTED_FEATURES:
                    if col not in chunk.columns:
                        chunk[col] = np.nan  # Add missing features with NaN
                
                # Extract features (X) and target (y)
                X = chunk[EXPECTED_FEATURES]
                y = chunk['responder_6'].astype(float)  # Ensure `y` is float
                
                # Handle missing values and scale features
                X = imputer.fit_transform(X) if i == 0 else imputer.transform(X)
                X = scaler.fit_transform(X) if i == 0 else scaler.transform(X)
                
                # Initialize the model once we know the feature size
                if model is None:
                    model = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3)
                
                # Incrementally train the model on the chunk
                model.partial_fit(X, y)
            
            except Exception as e:
                print(f"Error during preprocessing or training: {e}")
                continue  # Skip problematic chunks
        
        # Step 3: Save the trained model and preprocessing objects
        print("Saving model and preprocessing objects...")
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(imputer, 'models/imputer.pkl')
        joblib.dump(model, 'models/sgd_regressor.pkl')
        print("Model and preprocessing objects saved successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")