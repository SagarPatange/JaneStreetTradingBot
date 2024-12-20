#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:45:48 2024

@author: abdur-rahmanibn-bilalwaajid
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_features(df: pd.DataFrame, scaler=None, imputer=None):
    """
    Preprocess features: Impute missing values and scale data.
    """
    features = [col for col in df.columns if 'feature_' in col]

    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        df[features] = imputer.fit_transform(df[features])
    else:
        df[features] = imputer.transform(df[features])

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])

    return df, features, scaler, imputer

