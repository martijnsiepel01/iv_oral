import pandas as pd
import numpy as np
from config import MISSING_VALUE    

def encode(encoders, value, key):
    val = str(value).strip() if value not in [None, "", "-"] else MISSING_VALUE
    try:
        return float(encoders[key].transform([val])[0])
    except ValueError:
        raise ValueError(f"[ENCODE ERROR] Unknown label '{val}' for key '{key}' in encoder. "
                         f"Known classes: {list(encoders[key].classes_)}")


def normalize_date(dt_str):
    return pd.to_datetime(dt_str).normalize()

def safe_stats(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return [0.0, 0.0, 0.0]
    arr = np.array(vals, dtype=float)
    return [float(arr.min()), float(arr.mean()), float(arr.max())]