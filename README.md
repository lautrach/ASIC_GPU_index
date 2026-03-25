# GPU Cloud Rental Price Index

A price index for GPU cloud compute. Scrapes live listing data from the Shadeform aggregator API, cleans and standardizes it, then computes three indices that answer different questions about the GPU rental market.

**Product group tracked:** H100, H200, B200, A100, A100_80G, RTX5090

## Usage

```bash
pip install pandas plotly aiohttp

# 1. Fetch latest rental listings from Shadeform
python fetch_shadeform_rental_prices.py
#    → data/shadeform_rental_prices.csv

# 2. Build the index dashboard
python price_analysis.py
#    → output/gpu_rental_index.html

# Custom input/output paths
python price_analysis.py --csv data/shadeform_rental_prices.csv --output output/gpu_rental_index.html
```

## Data Pipeline

```
Shadeform API ──→ fetch_shadeform_rental_prices.py ──→ shadeform_rental_prices.csv
                                                              │
                                                      price_analysis.py
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                          ▼                   ▼                   ▼
                                    Index 1: Price      Index 2: VRAM       Index 3: Regional
                                    ($/GPU/hr)          ($/GB/hr)           (US/EU/APAC)
                                          │                   │                   │
                                          └───────────────────┼───────────────────┘
                                                              ▼
                                                  gpu_rental_index.html
```

### Data cleaning

Raw listings are one row per (provider × region × GPU config). Before indexing:

1. **Filter** CPU-only rows (`num_gpus == 0`)
2. **Compute unit price:** `price_per_gpu_hour = price_per_hour_usd / num_gpus`
3. **Deduplicate** exact matches on (provider, region, gpu_model, num_gpus, price)
4. **Classify regions** into US, EU, APAC via heuristic matching on country codes, US state abbreviations, and city names

## The Three Indices

### Index 1 — Headline Median Price ($/GPU/hr)

The core price signal. For each GPU model, we report the **weighted median** across providers, plus P25 and P75 to show the interquartile range (IQR).

**Why weighted average, not simple average?**

The raw data is one row per (provider × region). A provider with 15 regions would get 15× the influence of a provider with 1 region in a naive average — that's measuring geographic footprint, not price. So we first **average within each provider** to get one price per provider per model.

But then equal-weighting all providers has the opposite problem: a niche provider with a single listing in one city counts the same as a major cloud with hundreds of instances globally. We use **listing count as a weight** — a proxy for market footprint. Providers offering more configurations represent a larger share of actual supply, so they should carry proportionally more weight in the index.

**Why median instead of mean?**

With only 4–13 providers per GPU model, a single outlier (e.g., one provider at $5.99/hr for H100 when the rest are $2–3) would drag the mean significantly. The **median is manipulation-resistant** — no single provider can move it without collusion from at least half the market. The IQR alongside it shows how tight or loose the market is.

The dashboard displays this as a **box plot** so you can see the full distribution of provider-level prices, not just a summary statistic.

**Example output:**
```
gpu_model      p25   median      p75      iqr  n_providers  n_obs
  RTX5090   0.6625   0.6750   0.6875   0.0250            2      5
     A100   1.2900   1.3600   1.4500   0.1600            5     52
 A100_80G   1.3583   1.6731   1.8200   0.4617            9     72
     H100   2.0938   2.4800   3.0294   0.9356           13     71
     H200   2.6140   3.0000   3.3994   0.7854            7     32
     B200   4.3204   4.6363   4.8541   0.5337            4     15
```

### Index 2 — VRAM-Normalized Efficiency ($/GB/hr)

Different GPUs have different VRAM. Comparing a $1.29/hr A100 (40 GB) to a $1.65/hr A100_80G head-to-head is misleading — you're getting twice the memory. This index normalizes:

```
dollar_per_gb_hr = median_price_per_gpu_hour / vram_per_gpu_gb
```

Lower is better. This reveals that:
- A100_80G at $1.67/hr → **$0.021/GB-hr** (good value)
- A100 at $1.36/hr → **$0.032/GB-hr** (cheaper per hour, but worse per GB)
- H100 at $2.48/hr → **$0.037/GB-hr** (premium for compute, not memory-efficient)

Useful when your workload is VRAM-bound (large model inference, long-context LLMs).

### Index 3 — Regional Spread (US / EU / APAC)

The headline index is region-agnostic. This companion metric breaks it down geographically, reporting median + min–max per region:

```
H100 index: $2.48 | US: $1.66–$5.99 | EU: $1.95–$3.34 | APAC: $2.49–$2.99
H200 index: $3.00 | US: $2.25–$3.65 | EU: $2.45–$4.23
B200 index: $4.64 | US: $3.74–$5.29 | EU: $4.20–$5.63 | APAC: $4.99–$5.29
```

Region classification rules:
- **US:** `us-*` prefixes, US state codes (`TX`, `CA`), Canadian cities (North America bucket)
- **EU:** Country codes DE, FI, FR, GB, NL, NO, PL, IS
- **APAC:** Country codes JP, IN, SG, AU, IL

## Dashboard

The output `gpu_rental_index.html` is a self-contained Plotly HTML file (no server needed, just open in a browser). Four panels:

| Panel | Chart | What it shows |
|-------|-------|---------------|
| 1 | Indicator cards | Headline median $/GPU/hr + IQR per model |
| 2 | Box plot | Full provider price distribution per model |
| 3 | Bar chart | VRAM efficiency $/GB/hr (lower = better) |
| 4 | Grouped bar | Regional price comparison US / EU / APAC |

## Files

```
├── fetch_shadeform_rental_prices.py   # Async Shadeform API fetcher → CSV
├── price_analysis.py                  # Index computation + HTML dashboard
├── data/
│   ├── shadeform_rental_prices.csv         # Raw API output
│   └── shadeform_rental_prices_cleaned.csv # Deduplicated + unit price + geo
└── output/
    └── gpu_rental_index.html               # Interactive dashboard
```
