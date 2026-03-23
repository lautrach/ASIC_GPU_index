"""
Core formulas:
    PaybackDays_{e,t} = ASICPricePerTH_{e,t} / NetHashpricePerTHPerDay_{e,t}
    NetHashpricePerTHPerDay_{e,t} = Hashprice_t - ElectricityCostPerTHPerDay_{e,t}
    ASICPaybackIndex_{e,t} = 100 × PaybackDays_{e,t} / PaybackDays_{e,0}

"""

import asyncio
import logging
import os
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

BASE_URL = "https://api.hashrateindex.com/v1/hashrateindex"

TIER_EFFICIENCY_J_PER_TH = {
    "under19": 15,
    "19to25": 22,
    "25to38": 31.5,
    "38to68": 53,
    "above68": 80,
}

TIER_LABELS = {
    "under19": "<19 J/TH",
    "19to25": "19–25 J/TH",
    "25to38": "25–38 J/TH",
    "38to68": "38–68 J/TH",
    "above68": ">68 J/TH",
}

TIER_COLORS = {
    "under19": "#2ecc71",
    "19to25": "#3498db",
    "25to38": "#f39c12",
    "38to68": "#e74c3c",
    "above68": "#9b59b6",
}

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "asic_payback_index.html")


# ─── API Fetchers ────────────────────────────────────────────────────

async def fetch_asic_price_index(
    session: aiohttp.ClientSession, span: str = "1Y"
) -> list:
    url = f"{BASE_URL}/asic/price-index"
    params = {"currency": "USD", "span": span}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error(f"ASIC price index API error {resp.status}: {body[:200]}")
            return []
        result = await resp.json()
        return result.get("data", [])


async def fetch_hashprice(
    session: aiohttp.ClientSession, span: str = "1Y", bucket: str = "6H"
) -> list:
    url = f"{BASE_URL}/hashprice"
    params = {"currency": "USD", "hashunit": "THS", "span": span, "bucket": bucket}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error(f"Hashprice API error {resp.status}: {body[:200]}")
            return []
        result = await resp.json()
        return result.get("data", [])


async def fetch_electricity_price(
    session: aiohttp.ClientSession, span: str = "ALL"
) -> list:
    url = f"{BASE_URL}/energy/electricity-price-per-sector"
    params = {"span": span}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            body = await resp.text()
            logger.error(f"Electricity price API error {resp.status}: {body[:200]}")
            return []
        result = await resp.json()
        return result.get("data", [])


# ─── Calculation ─────────────────────────────────────────────────────

def calc_electricity_cost_per_th_day(
    efficiency_j_per_th: float, elec_price_usd_per_mwh: float
) -> float:
    """
    Convert electricity price to cost per TH/s per day.

    At 1 TH/s, power draw = efficiency (J/TH) = efficiency (W).
    Energy per day = efficiency_W × 24h = efficiency × 24 Wh.
    Convert to MWh: × 1e-6.
    Cost = energy_MWh × price_per_MWh.
    """
    return efficiency_j_per_th * 24 / 1_000_000 * elec_price_usd_per_mwh


def compute_payback_index(
    asic_raw: list, hashprice_raw: list, electricity_raw: list,
    sector: str = "industrial",
) -> pd.DataFrame:
    """
    Align the three data sources and compute PaybackDays + Index per tier.
    """
    # --- Build DataFrames ---
    df_asic = pd.DataFrame(asic_raw)
    df_asic["timestamp"] = pd.to_datetime(df_asic["timestamp"], utc=True)
    df_asic = df_asic.sort_values("timestamp").reset_index(drop=True)

    df_hp = pd.DataFrame(hashprice_raw)
    df_hp["timestamp"] = pd.to_datetime(df_hp["timestamp"], utc=True)
    df_hp = df_hp.sort_values("timestamp").reset_index(drop=True)
    df_hp = df_hp.rename(columns={"price": "hashprice"})

    df_elec = pd.DataFrame(electricity_raw)
    df_elec["timestamp"] = pd.to_datetime(df_elec["date"], utc=True)
    df_elec = df_elec.sort_values("timestamp").reset_index(drop=True)
    elec_col = sector  # "industrial", "commercial", or "residential"

    # --- Merge on nearest timestamp ---
    # Hashprice has higher frequency (6H) than ASIC (daily) and electricity (monthly).
    # Use ASIC timestamps as the base and merge_asof for the others.
    df = pd.merge_asof(
        df_asic[["timestamp"] + list(TIER_EFFICIENCY_J_PER_TH.keys())],
        df_hp[["timestamp", "hashprice"]],
        on="timestamp",
        direction="nearest",
    )
    df = pd.merge_asof(
        df,
        df_elec[["timestamp", elec_col]],
        on="timestamp",
        direction="nearest",
    )
    df = df.rename(columns={elec_col: "elec_price_mwh"})

    # --- Compute PaybackDays per tier ---
    records = []
    for tier, eff in TIER_EFFICIENCY_J_PER_TH.items():
        elec_cost = calc_electricity_cost_per_th_day(eff, df["elec_price_mwh"])
        net_hp = df["hashprice"] - elec_cost
        asic_price = df[tier].astype(float)

        payback = asic_price / net_hp
        # Set negative/invalid to NaN (happens when net hashprice ≤ 0)
        payback[net_hp <= 0] = float("nan")

        tier_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "tier": tier,
            "asic_price_per_th": asic_price,
            "hashprice": df["hashprice"],
            "elec_cost_per_th_day": elec_cost,
            "net_hashprice": net_hp,
            "payback_days": payback,
        })
        records.append(tier_df)

    result = pd.concat(records, ignore_index=True)

    # --- Compute Index (base = first valid PaybackDays per tier = 100) ---
    base_values = (
        result.dropna(subset=["payback_days"])
        .groupby("tier")["payback_days"]
        .first()
    )
    result["base_payback_days"] = result["tier"].map(base_values)
    result["index_value"] = 100 * result["payback_days"] / result["base_payback_days"]

    return result


# ─── HTML Chart Generation ───────────────────────────────────────────

def generate_html(df: pd.DataFrame, output_path: str) -> None:
    """Generate a self-contained HTML file with 4 Plotly charts."""

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "ASIC Payback Index (Base = 100)",
            "Payback Days by Efficiency Tier",
            "Hashprice ($/TH/s/day)",
            "Industrial Electricity Price ($/MWh)",
        ),
        row_heights=[0.3, 0.3, 0.2, 0.2],
    )

    tiers = list(TIER_EFFICIENCY_J_PER_TH.keys())

    # Chart 1: Payback Index
    for tier in tiers:
        tier_data = df[df["tier"] == tier].sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=tier_data["timestamp"],
                y=tier_data["index_value"],
                name=TIER_LABELS[tier],
                line=dict(color=TIER_COLORS[tier], width=2),
                legendgroup=tier,
                hovertemplate="%{x|%Y-%m-%d}<br>Index: %{y:.1f}<extra>" + TIER_LABELS[tier] + "</extra>",
            ),
            row=1, col=1,
        )

    # Chart 2: Payback Days
    for tier in tiers:
        tier_data = df[df["tier"] == tier].sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=tier_data["timestamp"],
                y=tier_data["payback_days"],
                name=TIER_LABELS[tier],
                line=dict(color=TIER_COLORS[tier], width=2),
                legendgroup=tier,
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Payback: %{y:.0f} days<extra>" + TIER_LABELS[tier] + "</extra>",
            ),
            row=2, col=1,
        )

    # Chart 3: Hashprice (single line, take first tier's data since hashprice is same for all)
    hp_data = df[df["tier"] == tiers[0]].sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=hp_data["timestamp"],
            y=hp_data["hashprice"],
            name="Hashprice",
            line=dict(color="#1abc9c", width=2),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.4f}/TH/s/day<extra>Hashprice</extra>",
        ),
        row=3, col=1,
    )

    # Chart 4: Electricity price (single line)
    elec_data = df[df["tier"] == tiers[0]].sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=elec_data["timestamp"],
            y=elec_data["elec_cost_per_th_day"],
            name="Elec Cost",
            line=dict(color="#e67e22", width=2),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.6f}/TH/s/day<extra>Electricity Cost</extra>",
        ),
        row=4, col=1,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="ASIC Payback Index Dashboard",
            font=dict(size=24),
        ),
        height=1200,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=2, col=1)
    fig.update_yaxes(title_text="$/TH/s/day", row=3, col=1)
    fig.update_yaxes(title_text="$/TH/s/day", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    # Add base=100 reference line
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    logger.info(f"HTML saved to {output_path}")


# ─── Main ────────────────────────────────────────────────────────────

async def main():
    api_key = os.environ.get("HRI_API_KEY")
    if not api_key:
        logger.error("Set the HRI_API_KEY environment variable")
        sys.exit(1)

    headers = {
        "X-Hi-Api-Key": api_key,
        "Accept": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        logger.info("Fetching data from Hashrate Index API...")

        asic_data, hashprice_data, electricity_data = await asyncio.gather(
            fetch_asic_price_index(session, span="1Y"),
            fetch_hashprice(session, span="1Y", bucket="6H"),
            fetch_electricity_price(session, span="ALL"),
        )

        if not asic_data:
            logger.error("No ASIC price index data received")
            sys.exit(1)
        if not hashprice_data:
            logger.error("No hashprice data received")
            sys.exit(1)
        if not electricity_data:
            logger.error("No electricity price data received")
            sys.exit(1)

        logger.info(
            f"Data received — ASIC: {len(asic_data)} points, "
            f"Hashprice: {len(hashprice_data)} points, "
            f"Electricity: {len(electricity_data)} points"
        )

        df = compute_payback_index(asic_data, hashprice_data, electricity_data)
        logger.info(f"Computed payback index: {len(df)} rows across {df['tier'].nunique()} tiers")

        generate_html(df, OUTPUT_FILE)
        logger.info(f"Done! Open {OUTPUT_FILE} in your browser.")


if __name__ == "__main__":
    asyncio.run(main())
