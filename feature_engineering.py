import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.copy()

    # 1) Handle timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour']      = df['timestamp'].dt.hour
        df['month']     = df['timestamp'].dt.month
        # (cyclical transforms could be added here)

    # 2) Identify numeric & categorical
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    # 3) Impute missing
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        mode = df[c].mode()
        df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else 'missing')

    # 4) Force numeric dtype on those columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # 5) Encode error_code as severity
    severity_map = {'E00': 0, 'E01': 1, 'E02': 2}
    df['error_severity'] = df['error_code']\
                             .map(severity_map)\
                             .fillna(0)\
                             .astype(int)

    # 6) One‐hot encode installation_type
    df = pd.get_dummies(df, columns=['installation_type'], drop_first=True)

    # 7) Ensure all source cols are numeric before deriving
    for col in [
        'temperature','humidity','irradiance','wind_speed',
        'module_temperature','voltage','current',
        'panel_age','maintenance_count','soiling_ratio'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 8) Derived interaction features
    df['temp_diff']         = df['module_temperature'] - df['temperature']
    df['power_output']      = df['voltage'] * df['current']
    df['age_maint_inter']   = df['panel_age'] * df['maintenance_count']
    df['soiling_irr_inter'] = df['soiling_ratio'] * df['irradiance']
    df['wind_temp_inter']   = df['wind_speed'] * df['module_temperature']

    # 9) Polynomial terms
    for col in ['temperature','humidity','irradiance']:
        df[f'{col}_sq'] = df[col] ** 2

    # 10) Rolling stats if timestamp exists
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        roll = df['irradiance'].rolling(3, min_periods=1).agg(['mean','std'])
        df['irradiance_roll3_mean'] = roll['mean']
        df['irradiance_roll3_std']  = roll['std'].fillna(0.0)

    # 11) Drop the original error_code (and timestamp if you like)
    df = df.drop(columns=['error_code'], errors='ignore')

    return df
