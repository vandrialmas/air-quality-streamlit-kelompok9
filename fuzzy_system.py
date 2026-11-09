import numpy as np

def compute_fuzzy_risk(co_value):
    co_val = np.clip(float(co_value), 0.0, 15.0)
    
    very_low  = max(0.0, min(1.0, (3.0 - co_val) / 3.0)) if co_val <= 3.0 else 0.0
    low       = max(0.0, min((co_val - 1.5)/2.0, (5.5 - co_val)/2.0)) if 1.5 <= co_val <= 5.5 else 0.0
    medium    = max(0.0, min((co_val - 4.0)/2.5, (9.0 - co_val)/2.5)) if 4.0 <= co_val <= 9.0 else 0.0
    high      = max(0.0, min((co_val - 7.5)/2.5, (12.5 - co_val)/2.5)) if 7.5 <= co_val <= 12.5 else 0.0
    very_high = max(0.0, min((co_val - 11.0)/2.0, 1.0)) if co_val >= 11.0 else 0.0
    
    risk_score = (very_low * 12.5 + low * 35.0 + medium * 60.0 + high * 78.0 + very_high * 93.0) / (
        very_low + low + medium + high + very_high + 1e-6
    )
    
    if risk_score < 22:
        return risk_score, "Good", "🟢", "Kualitas udara sangat baik. Aman untuk beraktivitas di luar ruangan."
    elif risk_score < 48:
        return risk_score, "Moderate", "🟡", "Kualitas udara dapat diterima. Kelompok sensitif disarankan mengurangi aktivitas fisik berat."
    elif risk_score < 68:
        return risk_score, "Unhealthy", "🟠", "Kualitas udara tidak sehat bagi kelompok sensitif. Gunakan masker jika di luar."
    elif risk_score < 86:
        return risk_score, "Very Unhealthy", "🔴", "Kualitas udara sangat tidak sehat. Hindari aktivitas luar ruangan."
    else:
        return risk_score, "Hazardous", "🟣", "Tingkat BAHAYA! Seluruh populasi wajib beraktivitas di dalam ruangan."