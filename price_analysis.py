"""
Indices:
  1. Rental GPU Median Price ($/GPU/hr) 
  2. VRAM-Normalized Efficiency ($/GB/hr)
  3. Regional Spread (US / EU / APAC breakdown)

Usage:
    python price_analysis.py
    python price_analysis.py --csv data/shadeform_rental_prices.csv
"""

import argparse
import logging
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CSV = "data/shadeform_rental_prices.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpu_rental_index.html")

# Product group to index
PRODUCT_GROUP = ["H100", "H200", "B200", "A100", "A100_80G", "RTX5090"]

MODEL_COLORS = {
    "H100": "#3498db",
    "H200": "#2ecc71",
    "B200": "#e74c3c",
    "A100": "#f39c12",
    "A100_80G": "#9b59b6",
    "RTX5090": "#1abc9c",
}


# ─── Region Classification ──────────────────────────────────────────

# Two-letter country prefixes that map to each geo bucket
_EU_PREFIXES = ("DE", "FI", "FR", "GB", "NL", "NO", "PL", "IS")
_APAC_PREFIXES = ("JP", "IN", "SG", "AU", "IL")
# Bare abbreviations that appear in the data
_US_BARE = {"atl", "ams"}  # "ams" is ambiguous but only 1 row — treat special
_APAC_BARE = {"syd2", "tyo4"}

# US state abbreviations (2-letter) that appear in region strings
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}


def classify_region(region: str) -> str:
    """Classify a Shadeform region string into US, EU, APAC, or Other."""
    r = region.strip()
    r_lower = r.lower()

    # # Bare abbreviations
    # if r_lower in {"ams"}:
    #     return "EU"
    # if r_lower in {"atl"}:
    #     return "US"
    # if r_lower in {"syd2", "tyo4"}:
    #     return "APAC"
    if r_lower in {"ams", "atl", "syd2", "tyo4"}:
        return None

    # "CANADA-2", "Montreal CA", "Calgary" etc.
    if "canada" in r_lower or "montreal" in r_lower or "calgary" in r_lower or "toronto" in r_lower:
        return "US"  # North America bucket

    # Standard format: "CC, City" — check first two chars
    prefix = r[:2].upper()

    # US-format regions often look like "City STATE" e.g. "Des Moines IA"
    # or use "us-" prefix like "us-midwest-3"
    if r_lower.startswith("us-") or r_lower.startswith("us,"):
        return "US"

    # Check for US state abbreviation at end: "Dallas TX", "New York NY"
    parts = r.split()
    if parts and len(parts[-1]) == 2 and parts[-1].upper() in _US_STATES:
        return "US"

    # Country-prefix format: "DE, Frankfurt"
    if prefix in _EU_PREFIXES:
        return "EU"
    if prefix in _APAC_PREFIXES:
        return "APAC"

    # Fallback heuristics
    if any(city in r_lower for city in [
        "kansas", "dallas", "chicago", "phoenix", "houston", "dulles",
        "salt lake", "san jose", "new york", "des moines",
    ]):
        return "US"

    return "Other"


# ─── Clean & Standardize ────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, add unit price, filter out non-GPU rows."""

    # Filter out CPU-only rows (num_gpus == 0)
    df = df[df["num_gpus"] > 0].copy()

    # Unit price: cost per single GPU per hour
    df["price_per_gpu_hour"] = (df["price_per_hour_usd"] / df["num_gpus"]).round(4)

    # Drop exact duplicate listings
    dedup_cols = ["provider", "region", "gpu_model", "num_gpus", "price_per_hour_usd"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    logger.info("Deduplication: %d → %d rows", before, len(df))

    # Add geo classification
    df["geo"] = df["region"].apply(classify_region)
    df = df.dropna(subset=["geo"])


    return df


# ─── Index 1: Headline Median Price ─────────────────────────────────

def compute_headline_index(
    df: pd.DataFrame, models: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # """
    # Compute P25, P50 (median), P75 of price_per_gpu_hour across provider
    # averages for each model.

    # Returns:
    #     headline: Summary DataFrame with p25, median, p75, iqr per model
    #     provider_avg: Per-provider average prices (used for box plot)
    # """
    # subset = df[df["gpu_model"].isin(models)]

    # provider_avg = (
    #     subset.groupby(["gpu_model", "provider"])["price_per_gpu_hour"]
    #     .mean()
    #     .reset_index()
    # )
    # headline = (
    #     provider_avg.groupby("gpu_model")["price_per_gpu_hour"]
    #     .quantile([0.25, 0.50, 0.75])
    #     .unstack()
    #     .rename(columns={0.25: "p25", 0.50: "median", 0.75: "p75"})
    #     .reset_index()
    # )
    # headline["iqr"] = headline["p75"] - headline["p25"]
    # headline["n_providers"] = (
    #     provider_avg.groupby("gpu_model")["provider"].count().values
    # )
    # headline["n_obs"] = (
    #     subset.groupby("gpu_model")["price_per_gpu_hour"].count().values
    # )

    # headline = headline.sort_values("median").reset_index(drop=True)
    # logger.info("Headline index:\n%s", headline.to_string(index=False))
    # return headline, provider_avg

    """
    Compute weighted P25, P50 (median), P75 of price_per_gpu_hour.
    
    Weights each provider by their number of listings (proxy for market
    footprint). Providers with more regions/configs get more influence.

    Returns:
        headline: Summary DataFrame with p25, median, p75, iqr per model
        provider_avg: Per-provider average prices with weights
    """
    subset = df[df["gpu_model"].isin(models)]

    # Provider's average price AND their listing count as weight
    provider_avg = (
        subset.groupby(["gpu_model", "provider"])
        .agg(
            price_per_gpu_hour=("price_per_gpu_hour", "mean"),
            n_listings=("price_per_gpu_hour", "count"),
        )
        .reset_index()
    )

    # Normalize weights within each model (sum to 1)
    provider_avg["weight"] = provider_avg.groupby("gpu_model")[
        "n_listings"
    ].transform(lambda x: x / x.sum())

    # Weighted quantiles per model
    def weighted_quantiles(group, quantiles=[0.25, 0.50, 0.75]):
        sorted_group = group.sort_values("price_per_gpu_hour")
        cumw = sorted_group["weight"].cumsum()
        results = {}
        for q in quantiles:
            idx = cumw.searchsorted(q)
            idx = min(idx, len(sorted_group) - 1)
            results[q] = sorted_group["price_per_gpu_hour"].iloc[idx]
        return pd.Series(results)

    headline = (
        provider_avg.groupby("gpu_model")
        .apply(weighted_quantiles, include_groups=False)
        .rename(columns={0.25: "p25", 0.50: "median", 0.75: "p75"})
        .reset_index()
    )
    headline["iqr"] = headline["p75"] - headline["p25"]
    headline["n_providers"] = (
        provider_avg.groupby("gpu_model")["provider"].count().values
    )
    headline["n_obs"] = (
        subset.groupby("gpu_model")["price_per_gpu_hour"].count().values
    )

    headline = headline.sort_values("median").reset_index(drop=True)
    logger.info("Headline index:\n%s", headline.to_string(index=False))
    return headline, provider_avg


# ─── Index 2: VRAM-Normalized Efficiency ─────────────────────────────

def compute_vram_efficiency(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compute $/GB/hr = median_price_per_gpu_hour / vram_per_gpu_gb."""
    subset = df[df["gpu_model"].isin(models)]

    agg = (
        subset.groupby("gpu_model")
        .agg(
            median_price=("price_per_gpu_hour", "median"),
            vram_gb=("vram_per_gpu_gb", "first"),
        )
        .reset_index()
    )
    agg["dollar_per_gb_hr"] = (agg["median_price"] / agg["vram_gb"]).round(6)

    # Sort by efficiency (lower is better)
    agg = agg.sort_values("dollar_per_gb_hr").reset_index(drop=True)
    logger.info("VRAM efficiency:\n%s", agg.to_string(index=False))
    return agg


# ─── Index 3: Regional Spread ───────────────────────────────────────

def compute_regional_spread(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """
    For each (model, geo_bucket), report median, min, max of price_per_gpu_hour.
    """
    subset = df[df["gpu_model"].isin(models)]

    spread = (
        subset.groupby(["gpu_model", "geo"])["price_per_gpu_hour"]
        .agg(median="median", min="min", max="max", n="count")
        .reset_index()
    )
    spread = spread.sort_values(["gpu_model", "geo"]).reset_index(drop=True)
    logger.info("Regional spread:\n%s", spread.to_string(index=False))
    return spread


# ─── HTML Dashboard ─────────────────────────────────────────────────

def generate_html(
    headline_df: pd.DataFrame,
    vram_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    provider_avg_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate a self-contained HTML dashboard with 4 panels."""

    n_models = len(headline_df)

    fig = make_subplots(
        rows=4, cols=n_models,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        specs=[
            # Row 1: indicator cards (one per model)
            [{"type": "indicator"}] * n_models,
            # Row 2: single bar chart spanning all cols
            [{"type": "xy", "colspan": n_models}] + [None] * (n_models - 1),
            # Row 3: VRAM efficiency bar chart
            [{"type": "xy", "colspan": n_models}] + [None] * (n_models - 1),
            # Row 4: regional heatmap table
            [{"type": "xy", "colspan": n_models}] + [None] * (n_models - 1),
        ],
        row_heights=[0.18, 0.28, 0.28, 0.26],
        subplot_titles=[
            *[f"" for _ in range(n_models)],  # indicator titles set via trace
            "", "", "",
        ],
    )

    # ── Panel 1: Indicator Cards ──────────────────────────────────
    for i, row in headline_df.iterrows():
        model = row["gpu_model"]
        color = MODEL_COLORS.get(model, "#95a5a6")
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=row["median"],
                number=dict(
                    prefix="$",
                    suffix="/hr",
                    font=dict(size=32, color=color),
                    valueformat=".2f",
                ),
                title=dict(
                    text=(
                        f"<b>{model}</b><br>"
                        f"<span style='font-size:12px;color:#aaa'>"
                        f"IQR: ${row['p25']:.2f}–${row['p75']:.2f}  "
                        f"({int(row['n_obs'])} obs)</span>"
                    ),
                    font=dict(size=16, color="white"),
                ),
            ),
            row=1, col=i + 1,
        )

    # ── Panel 2: Box Plot showing median + quantile distribution ───
    models_sorted = headline_df["gpu_model"].tolist()

    for model in models_sorted:
        color = MODEL_COLORS.get(model, "#95a5a6")
        model_prices = provider_avg_df[
            provider_avg_df["gpu_model"] == model
        ]["price_per_gpu_hour"].tolist()

        fig.add_trace(
            go.Box(
                y=model_prices,
                name=model,
                marker_color=color,
                line_color=color,
                boxmean=False,  # show only median line, not mean
                boxpoints="all",
                jitter=0.4,
                pointpos=-1.5,
                marker=dict(size=5, opacity=0.6),
                showlegend=False,
                hovertemplate=(
                    "<b>" + model + "</b><br>"
                    "$%{y:.2f}/GPU/hr<br>"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="$/hr", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)

    # Add annotation title for panel 2
    fig.add_annotation(
        text="<b>GPU Rental Price Index, $/hr</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.78, showarrow=False,
        font=dict(size=15, color="white"),
    )

    # ── Panel 3: VRAM Efficiency Bar Chart ────────────────────────
    vram_models = vram_df["gpu_model"].tolist()
    vram_vals = vram_df["dollar_per_gb_hr"].tolist()
    vram_colors = [MODEL_COLORS.get(m, "#95a5a6") for m in vram_models]
    vram_labels = [
        f"{m} ({int(v)}GB)" for m, v in zip(vram_models, vram_df["vram_gb"])
    ]

    fig.add_trace(
        go.Bar(
            x=vram_labels,
            y=vram_vals,
            marker_color=vram_colors,
            text=[f"${v:.4f}" for v in vram_vals],
            textposition="outside",
            textfont=dict(color="white", size=11),
            showlegend=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "VRAM Efficiency: $%{y:.4f}/GB/hr<br>"
                "<extra></extra>"
            ),
        ),
        row=3, col=1,
    )
    fig.update_yaxes(title_text="$/GB/hr (lower = better value)", row=3, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)

    fig.add_annotation(
        text="<b>Index 2 — VRAM-Normalized Efficiency ($/GB/hr)</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.48, showarrow=False,
        font=dict(size=15, color="white"),
    )

    # ── Panel 4: Regional Spread Grouped Bar ──────────────────────
    geo_order = ["US", "EU", "APAC", "Other"]
    geo_colors = {"US": "#3498db", "EU": "#2ecc71", "APAC": "#e74c3c", "Other": "#95a5a6"}

    # Get models in our product group that have regional data
    regional_models = [m for m in PRODUCT_GROUP if m in regional_df["gpu_model"].values]

    for geo in geo_order:
        geo_data = regional_df[regional_df["geo"] == geo]
        if geo_data.empty:
            continue

        y_vals = []
        x_labels = []
        hover_texts = []
        for model in regional_models:
            row_data = geo_data[geo_data["gpu_model"] == model]
            if row_data.empty:
                y_vals.append(0)
                x_labels.append(model)
                hover_texts.append("No data")
            else:
                r = row_data.iloc[0]
                y_vals.append(r["median"])
                x_labels.append(model)
                hover_texts.append(
                    f"<b>{model} — {geo}</b><br>"
                    f"Median: ${r['median']:.2f}<br>"
                    f"Range: ${r['min']:.2f}–${r['max']:.2f}<br>"
                    f"N: {int(r['n'])}"
                )

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_vals,
                name=geo,
                marker_color=geo_colors.get(geo, "#95a5a6"),
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=4, col=1,
        )

    fig.update_yaxes(title_text="$/GPU/hr by Region", row=4, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)
    fig.update_layout(barmode="group")

    fig.add_annotation(
        text="<b>Index 3 — Regional Price Spread</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.20, showarrow=False,
        font=dict(size=15, color="white"),
    )

    # ── Global Layout ─────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="GPU Cloud Rental Price Index Dashboard",
            font=dict(size=24),
        ),
        height=1400,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.03,
            xanchor="center",
            x=0.5,
        ),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    logger.info("HTML dashboard saved to %s", output_path)


# ─── Main ────────────────────────────────────────────────────────────

def main(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    df = clean(df)

    # Filter to product group
    available_models = [m for m in PRODUCT_GROUP if m in df["gpu_model"].values]
    logger.info("Product group models found: %s", available_models)

    if not available_models:
        logger.error("No product group models found in data!")
        return

    # Compute the three indices
    headline_df, provider_avg_df = compute_headline_index(df, available_models)
    vram_df = compute_vram_efficiency(df, available_models)
    regional_df = compute_regional_spread(df, available_models)

    # Print regional summary in one-liner format
    print("\n=== Regional Spread Summary ===\n")
    for model in available_models:
        parts = [model]
        model_data = regional_df[regional_df["gpu_model"] == model]
        headline_row = headline_df[headline_df["gpu_model"] == model]
        if not headline_row.empty:
            parts[0] += f" index: ${headline_row.iloc[0]['median']:.2f}"
        for geo in ["US", "EU", "APAC"]:
            geo_row = model_data[model_data["geo"] == geo]
            if not geo_row.empty:
                r = geo_row.iloc[0]
                if r["min"] == r["max"]:
                    parts.append(f"{geo}: ${r['median']:.2f}")
                else:
                    parts.append(f"{geo}: ${r['min']:.2f}–${r['max']:.2f}")
        print(" | ".join(parts))

    # Generate HTML dashboard
    generate_html(headline_df, vram_df, regional_df, provider_avg_df, output_path)

    # Save cleaned data
    out_clean = csv_path.replace(".csv", "_cleaned.csv")
    df.to_csv(out_clean, index=False)
    logger.info("Saved cleaned data to %s", out_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Rental Price Index Dashboard")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output HTML path")
    args = parser.parse_args()
    main(args.csv, args.output)
