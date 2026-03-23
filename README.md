GPU Price Index — Standalone Script

Fetches GPU mining models(power and hashrate) from WhatToMine API, merges with WhatToMine 
price CSV, computes a GPU Price Index per efficiency tier, and outputs
a self-contained HTML file with interactive Plotly charts. 
The electricity information takes industrial price of 0.0862  # $/kWh — US Industrial eia.gov (2026).

Core formulas:
    daily_elec_cost_i = power_w_i × 24 / 1000 × elec_price_per_kwh
    cost_metric_i     = (gpu_price_i + daily_elec_cost_i) / hashrate_mhs_i
    tier_avg_t        = Σ(cost_metric_i × hashrate_i) / Σ(hashrate_i)   [hashrate-weighted]
    GPU_PriceIndex    = 100 × tier_avg_t / tier_avg_0

Usage:
    export WTM_API_TOKEN="your-whattomine-token"
    python -m src.gpu_price_index
    python -m src.gpu_price_index --elec-price 0.0862
"""
