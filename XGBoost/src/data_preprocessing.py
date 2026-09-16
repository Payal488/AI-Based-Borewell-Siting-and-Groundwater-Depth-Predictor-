import os
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'rainfall', 't2m', 't2m_max', 't2m_min', 'month',
    'latitude', 'longitude', 'wellDepth',
    'wellAquiferType_encoded', 'State_encoded', 'District_encoded',
    'temp_range'
]

def clean_column_names(df):
    rename_dict = {}
    for col in df.columns:
        c = col.strip()
        if 'kmnpnn' in c or ('2m' in c and c not in ['t2m_max', 't2m_min']):
            rename_dict[col] = 't2m'
        else:
            rename_dict[col] = c
    return df.rename(columns=rename_dict)

def load_and_preprocess_data(csv_path="data/groundwater_data.csv", depth_threshold_ratio=0.80):
    df = pd.read_csv(csv_path)
    df = clean_column_names(df).drop_duplicates()
    df['gw_depth'] = df['target'].abs()
    
    for col in ['rainfall', 't2m', 't2m_max', 't2m_min', 'wellDepth', 'gw_depth']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
        
    for col in ['wellAquiferType_encoded', 'State_encoded', 'District_encoded']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0).astype(int)
        
    df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(6).astype(int)
    df['temp_range'] = (df['t2m_max'] - df['t2m_min']).clip(lower=0)
    
    # Derived success label
    df['borewell_success'] = (
        (df['gw_depth'] < (df['wellDepth'] * depth_threshold_ratio)) & 
        (df['gw_depth'] <= 25.0)
    ).astype(int)
    
    return df[FEATURE_COLS], df['gw_depth'], df['borewell_success'], df
