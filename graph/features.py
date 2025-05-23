import pandas as pd
import numpy as np
    
def encode(label_encoders, value, key):
    return float(label_encoders[key].transform([str(value)])[0])

def normalize_date(dt_str):
    return pd.to_datetime(dt_str).normalize()

def safe_stats(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return [0.0, 0.0, 0.0]
    arr = np.array(vals, dtype=float)
    return [float(arr.min()), float(arr.mean()), float(arr.max())]