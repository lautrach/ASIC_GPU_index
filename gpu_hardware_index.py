"""
GPU Price Index — Standalone Script

Fetches GPU mining specs from WhatToMine API, merges with user-maintained
price CSV, computes a GPU Price Index per efficiency tier, and outputs
a self-contained HTML file with interactive Plotly charts.

Core formulas:
    daily_elec_cost_i = power_w_i × 24 / 1000 × elec_price_per_kwh
    cost_metric_i     = (gpu_price_i + daily_elec_cost_i) / hashrate_mhs_i
    tier_avg_t        = Σ(cost_metric_i × hashrate_i) / Σ(hashrate_i)   [hashrate-weighted]
    GPU_PriceIndex    = 100 × tier_avg_t / tier_avg_0

Usage:
    export WTM_API_TOKEN="your-whattomine-token"
    python -m src.gpu_price_index
    python -m src.gpu_price_index --elec-price 0.07
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys

import aiohttp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────

WTM_BASE_URL = "https://whattomine.com/api/v1"
ALGORITHM = "Ethash"  # Filter for Ethash/Etchash mining

# Efficiency tiers in MH/W (megahash per watt) for Ethash
GPU_TIER_BOUNDS = {
    "latest_gen": (0.40, float("inf")),   # ≥0.40 MH/W
    "current_gen": (0.30, 0.40),          # 0.30–0.40 MH/W
    "mid_gen": (0.20, 0.30),              # 0.20–0.30 MH/W
    "old_gen": (0.0, 0.20),               # <0.20 MH/W
}

GPU_TIER_LABELS = {
    "latest_gen": "≥0.40 MH/W (Latest Gen)",
    "current_gen": "0.30–0.40 MH/W (Current Gen)",
    "mid_gen": "0.20–0.30 MH/W (Mid Gen)",
    "old_gen": "<0.20 MH/W (Old Gen)",
}

GPU_TIER_COLORS = {
    "latest_gen": "#2ecc71",   # green
    "current_gen": "#3498db",  # blue
    "mid_gen": "#f39c12",      # orange
    "old_gen": "#e74c3c",      # red
}

DEFAULT_ELEC_PRICE_KWH = 0.0862  # $/kWh — US Industrial eia.gov (2026)
CSV_PATH = "data/gpu_prices.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpu_price_index.html")


# ─── Name Normalization ─────────────────────────────────────────────

def normalize_gpu_name(name: str) -> str:
    """
    Normalize GPU model name for matching between WhatToMine API and CSV.

    Examples:
        "NVIDIA GeForce RTX 4090"  → "rtx 4090"
        "RTX 3060 Ti"              → "rtx 3060 ti"
        "AMD Radeon RX 580 8GB"    → "rx 580 8gb"
    """
    name = name.lower().strip()
    # Remove common prefixes
    for prefix in ["nvidia geforce ", "nvidia ", "geforce ", "amd radeon ", "amd ", "radeon "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ─── API Fetcher ─────────────────────────────────────────────────────

async def fetch_gpu_specs(session: aiohttp.ClientSession) -> tuple[list, pd.DataFrame]:
    """
    Fetch GPU specs from WhatToMine API.

    GET /api/v1/gpus returns:
    [
      {
        "id": 1,
        "name": "NVIDIA GeForce RTX 4090",
        "release_date": "2022-10-12",
        "algorithms": [
          {"name": "Ethash", "hashrate": 125000000, "power": 350},
          ...
        ]
      },
      ...
    ]

    Hashrate is in H/s — we convert to MH/s (÷ 1e6).
    """
    url = f"{WTM_BASE_URL}/gpus"
    async with session.get(url) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error(f"WhatToMine GPU API error {resp.status}: {body[:200]}")
            return [], pd.DataFrame()

        gpus_raw = await resp.json()

    records = []
    for gpu in gpus_raw:
        gpu_name = gpu.get("name", "")
        release_date = gpu.get("release_date")
        algorithms = gpu.get("algorithms", [])

        # Find Ethash algorithm entry
        ethash = None
        for algo in algorithms:
            if algo.get("name", "").lower() in ("ethash", "etchash"):
                ethash = algo
                break

        if ethash is None:
            continue  # Skip GPUs without Ethash support

        try:
            hashrate_hs = float(ethash.get("hashrate", 0))
            power_w = float(ethash.get("power", 0))
        except (TypeError, ValueError):
            continue

        if hashrate_hs <= 0 or power_w <= 0:
            continue

        hashrate_mhs = hashrate_hs / 1e6  # Convert H/s → MH/s

        records.append({
            "wtm_id": gpu.get("id"),
            "gpu_name_raw": gpu_name,
            "gpu_name_norm": normalize_gpu_name(gpu_name),
            "release_date": release_date,
            "hashrate_mhs": hashrate_mhs,
            "power_w": power_w,
        })

    df = pd.DataFrame(records)
    logger.info(f"WhatToMine: {len(df)} GPUs with Ethash support fetched")
    return gpus_raw, df


# ─── CSV Loader ──────────────────────────────────────────────────────

def load_gpu_prices(csv_path: str) -> pd.DataFrame:
    """Load GPU price data from CSV."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    required = {"gpu_model", "price_usd", "date"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"CSV missing required columns: {missing}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["gpu_name_norm"] = df["gpu_model"].apply(normalize_gpu_name)
    df["price_usd"] = df["price_usd"].astype(float)

    logger.info(f"CSV: {len(df)} price entries loaded from {csv_path}")
    return df


# ─── Merge & Compute ────────────────────────────────────────────────

def assign_tier(efficiency_mhw: float) -> str:
    """Assign a GPU to an efficiency tier based on MH/W."""
    for tier, (lo, hi) in GPU_TIER_BOUNDS.items():
        if lo <= efficiency_mhw < hi:
            return tier
    return "old_gen"


def merge_specs_and_prices(
    specs_df: pd.DataFrame, prices_df: pd.DataFrame, elec_price_kwh: float
) -> pd.DataFrame:
    """
    Merge WhatToMine specs with CSV prices on normalized GPU name.
    Compute efficiency, electricity cost, and cost metric.
    """
    if specs_df.empty or prices_df.empty:
        logger.error("Cannot merge: specs or prices DataFrame is empty")
        return pd.DataFrame()

    # Merge on normalized name
    merged = prices_df.merge(
        specs_df[["gpu_name_norm", "hashrate_mhs", "power_w"]],
        on="gpu_name_norm",
        how="inner",
    )

    # Log unmatched models from CSV
    unmatched = set(prices_df["gpu_name_norm"]) - set(specs_df["gpu_name_norm"])
    if unmatched:
        logger.warning(
            f"{len(unmatched)} GPU models from CSV not found in WhatToMine: "
            f"{sorted(unmatched)}"
        )

    matched = set(prices_df["gpu_name_norm"]) & set(specs_df["gpu_name_norm"])
    logger.info(f"Matched {len(matched)} GPU models between CSV and WhatToMine")

    if merged.empty:
        logger.error("No GPU models matched between CSV and WhatToMine specs")
        return pd.DataFrame()

    # Compute derived columns
    merged["efficiency_mhw"] = merged["hashrate_mhs"] / merged["power_w"]
    merged["daily_elec_cost"] = merged["power_w"] * 24 / 1000 * elec_price_kwh
    merged["cost_metric"] = (
        (merged["price_usd"] + merged["daily_elec_cost"]) / merged["hashrate_mhs"]
    )
    merged["tier"] = merged["efficiency_mhw"].apply(assign_tier)

    logger.info(
        f"Merged dataset: {len(merged)} rows, "
        f"tiers: {merged['tier'].value_counts().to_dict()}"
    )
    return merged


def compute_gpu_price_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hashrate-weighted GPU Price Index per tier.

    For each (date, tier):
        weighted_cost = Σ(cost_metric_i × hashrate_i) / Σ(hashrate_i)
    Then normalize: index = 100 × weighted_cost / base_weighted_cost
    """
    records = []
    for (date, tier), group in df.groupby(["date", "tier"]):
        total_hashrate = group["hashrate_mhs"].sum()
        weighted_cost = (
            (group["cost_metric"] * group["hashrate_mhs"]).sum() / total_hashrate
        )
        records.append({
            "date": date,
            "tier": tier,
            "weighted_cost_metric": weighted_cost,
            "gpu_count": len(group),
            "total_hashrate_mhs": total_hashrate,
        })

    index_df = pd.DataFrame(records)

    if index_df.empty:
        return index_df

    # Normalize to index = 100 at first date per tier
    base_values = (
        index_df.sort_values("date")
        .groupby("tier")["weighted_cost_metric"]
        .first()
    )
    index_df["base_cost"] = index_df["tier"].map(base_values)
    index_df["index_value"] = 100 * index_df["weighted_cost_metric"] / index_df["base_cost"]

    return index_df


# ─── HTML Chart Generation ───────────────────────────────────────────

def generate_html(
    index_df: pd.DataFrame, detail_df: pd.DataFrame, output_path: str
) -> None:
    """Generate a self-contained HTML file with 4 Plotly charts."""

    n_dates = index_df["date"].nunique()
    is_timeseries = n_dates >= 2

    tiers = list(GPU_TIER_BOUNDS.keys())

    if is_timeseries:
        # ─── Time-series mode: 4 rows of line/scatter charts ─────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.07,
            subplot_titles=(
                "GPU Price Index (Base = 100)",
                "Weighted Cost Metric by Tier ($/MH/s)",
                "Efficiency vs Cost (per GPU)",
                "Price vs Hashrate (per GPU)",
            ),
            row_heights=[0.28, 0.22, 0.25, 0.25],
        )
        scatter_start_row = 3

        # Chart 1: Index line chart
        for tier in tiers:
            tier_data = index_df[index_df["tier"] == tier].sort_values("date")
            if tier_data.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=tier_data["date"],
                    y=tier_data["index_value"],
                    name=GPU_TIER_LABELS[tier],
                    line=dict(color=GPU_TIER_COLORS[tier], width=2),
                    legendgroup=tier,
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>Index: %{y:.1f}<extra>"
                        + GPU_TIER_LABELS[tier] + "</extra>"
                    ),
                ),
                row=1, col=1,
            )
        fig.add_hline(
            y=100, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1
        )

        # Chart 2: Cost metric line chart
        for tier in tiers:
            tier_data = index_df[index_df["tier"] == tier].sort_values("date")
            if tier_data.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=tier_data["date"],
                    y=tier_data["weighted_cost_metric"],
                    name=GPU_TIER_LABELS[tier],
                    line=dict(color=GPU_TIER_COLORS[tier], width=2),
                    legendgroup=tier,
                    showlegend=False,
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>$%{y:.2f}/MH/s<extra>"
                        + GPU_TIER_LABELS[tier] + "</extra>"
                    ),
                ),
                row=2, col=1,
            )

    else:
        # ─── Snapshot mode: indicator cards + 2 scatter charts ───
        n_tiers = len(tiers)
        fig = make_subplots(
            rows=3, cols=n_tiers,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[
                [{"type": "indicator"}] * n_tiers,
                [{"type": "xy", "colspan": n_tiers}] + [None] * (n_tiers - 1),
                [{"type": "xy", "colspan": n_tiers}] + [None] * (n_tiers - 1),
            ],
            subplot_titles=(
                *[GPU_TIER_LABELS[t] for t in tiers],
                *[""] * (n_tiers * 2),
            ),
            row_heights=[0.20, 0.40, 0.40],
        )
        scatter_start_row = 2

        # Row 1: Indicator cards showing index value + cost metric
        for i, tier in enumerate(tiers):
            tier_data = index_df[index_df["tier"] == tier]
            if tier_data.empty:
                continue

            cost_val = tier_data["weighted_cost_metric"].values[0]
            gpu_count = int(tier_data["gpu_count"].values[0])

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=cost_val,
                    number=dict(
                        prefix="$",
                        suffix="/MH/s",
                        font=dict(size=36, color=GPU_TIER_COLORS[tier]),
                        valueformat=".2f",
                    ),
                    title=dict(
                        text=f"{gpu_count} GPUs",
                        font=dict(size=14, color="#aaa"),
                    ),
                ),
                row=1, col=i + 1,
            )

        # Add section titles as annotations for scatter rows
        fig.add_annotation(
            text="Efficiency vs Cost (per GPU)",
            xref="paper", yref="paper",
            x=0.5, y=0.72, showarrow=False,
            font=dict(size=16, color="white"),
        )
        fig.add_annotation(
            text="Price vs Hashrate (per GPU)",
            xref="paper", yref="paper",
            x=0.5, y=0.35, showarrow=False,
            font=dict(size=16, color="white"),
        )

    # ─── Scatter: Efficiency vs Cost ────────────────────────────
    latest_date = detail_df["date"].max()
    scatter_df = detail_df[detail_df["date"] == latest_date]
    scatter_row_eff = scatter_start_row
    scatter_row_hp = scatter_start_row + 1

    # In snapshot mode, scatter plots span all columns (col=1 only, colspan via specs)
    # In timeseries mode, they're in the standard single-column layout
    scatter_col = 1

    for tier in tiers:
        tier_data = scatter_df[scatter_df["tier"] == tier]
        if tier_data.empty:
            continue

        # Efficiency vs Cost
        fig.add_trace(
            go.Scatter(
                x=tier_data["efficiency_mhw"],
                y=tier_data["cost_metric"],
                mode="markers+text",
                name=GPU_TIER_LABELS[tier],
                marker=dict(
                    color=GPU_TIER_COLORS[tier],
                    size=tier_data["hashrate_mhs"].clip(upper=150) / 3 + 6,
                    opacity=0.8,
                ),
                text=tier_data["gpu_model"],
                textposition="top center",
                textfont=dict(size=9),
                legendgroup=tier,
                showlegend=not is_timeseries,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Efficiency: %{x:.3f} MH/W<br>"
                    "Cost: $%{y:.2f}/MH/s<br>"
                    "<extra>" + GPU_TIER_LABELS[tier] + "</extra>"
                ),
            ),
            row=scatter_row_eff, col=scatter_col,
        )

        # Price vs Hashrate
        fig.add_trace(
            go.Scatter(
                x=tier_data["hashrate_mhs"],
                y=tier_data["price_usd"],
                mode="markers+text",
                name=GPU_TIER_LABELS[tier],
                marker=dict(
                    color=GPU_TIER_COLORS[tier],
                    size=10,
                    opacity=0.8,
                ),
                text=tier_data["gpu_model"],
                textposition="top center",
                textfont=dict(size=9),
                legendgroup=tier,
                showlegend=False,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Hashrate: %{x:.1f} MH/s<br>"
                    "Price: $%{y:.0f}<br>"
                    "<extra>" + GPU_TIER_LABELS[tier] + "</extra>"
                ),
            ),
            row=scatter_row_hp, col=scatter_col,
        )

    # ─── Layout ──────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="GPU Price Index Dashboard (Ethash/ETC Mining)",
            font=dict(size=24),
        ),
        height=1400 if is_timeseries else 1200,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    if is_timeseries:
        fig.update_yaxes(title_text="Index", row=1, col=1)
        fig.update_yaxes(title_text="$/MH/s", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_yaxes(title_text="$/MH/s", row=scatter_row_eff, col=1)
    fig.update_xaxes(title_text="Efficiency (MH/W)", row=scatter_row_eff, col=1)
    fig.update_yaxes(title_text="Price ($)", row=scatter_row_hp, col=1)
    fig.update_xaxes(title_text="Hashrate (MH/s)", row=scatter_row_hp, col=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    logger.info(f"HTML saved to {output_path}")


# ─── Main ────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="GPU Price Index for Ethash/ETC mining"
    )
    parser.add_argument(
        "--csv", default=CSV_PATH,
        help=f"Path to GPU prices CSV (default: {CSV_PATH})"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output HTML path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--elec-price", type=float, default=DEFAULT_ELEC_PRICE_KWH,
        help=(
            f"Electricity price in $/kWh (default: {DEFAULT_ELEC_PRICE_KWH}). "
            "Reference rates — Industrial: $0.0845, Commercial: $0.1341, Residential: $0.1730"
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save raw WhatToMine API response to data/wtm_gpu_specs.json",
    )
    args = parser.parse_args()

    # Check API token
    api_token = os.environ.get("WTM_API_TOKEN")
    if not api_token:
        logger.error("Set the WTM_API_TOKEN environment variable")
        sys.exit(1)

    # Fetch GPU specs from WhatToMine
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        logger.info("Fetching GPU specs from WhatToMine API...")
        gpus_raw, specs_df = await fetch_gpu_specs(session)

    if args.debug and gpus_raw:
        debug_path = "data/wtm_gpu_specs.json"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w") as f:
            json.dump(gpus_raw, f, indent=2)
        logger.info(f"Debug: raw API response saved to {debug_path}")

    if specs_df.empty:
        logger.error("No GPU specs received from WhatToMine")
        sys.exit(1)

    # Load prices from CSV
    prices_df = load_gpu_prices(args.csv)
    if prices_df.empty:
        logger.error("No price data loaded from CSV")
        sys.exit(1)

    # Merge and compute
    merged_df = merge_specs_and_prices(specs_df, prices_df, args.elec_price)
    if merged_df.empty:
        logger.error("No data after merging specs and prices")
        sys.exit(1)

    index_df = compute_gpu_price_index(merged_df)
    logger.info(
        f"Computed index: {len(index_df)} rows, "
        f"{index_df['tier'].nunique()} tiers, "
        f"{index_df['date'].nunique()} dates"
    )

    # Generate HTML
    generate_html(index_df, merged_df, args.output)
    logger.info(f"Done! Open {args.output} in your browser.")


if __name__ == "__main__":
    asyncio.run(main())
