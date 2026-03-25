"""
Shadeform GPU Rental Prices — Data Collection Script

Fetches GPU cloud rental instance data from the Shadeform API,
flattens the nested JSON into a tabular format, and saves to CSV.

Data includes: provider, region, GPU model, count, VRAM, hourly price,
availability status, and interconnect type.

Usage:
    python shadeform_rental_prices.py
    python shadeform_rental_prices.py --output data/shadeform_rental_prices.csv
"""

import argparse
import asyncio
import logging
from datetime import date

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────

SHADEFORM_API_URL = "https://api.shadeform.ai/v1/instances/types"
DEFAULT_OUTPUT = "data/shadeform_rental_prices.csv"


# ─── Fetch ───────────────────────────────────────────────────────────

async def fetch_instance_types(session: aiohttp.ClientSession) -> list:
    """Fetch all GPU instance types from the Shadeform API."""
    logger.info("Fetching instance types from Shadeform API …")
    async with session.get(SHADEFORM_API_URL) as resp:
        if resp.status != 200:
            logger.error("Shadeform API returned status %d", resp.status)
            return []
        data = await resp.json()

    instances = data.get("instance_types", [])
    logger.info("Fetched %d instance types", len(instances))
    return instances


# ─── Flatten ─────────────────────────────────────────────────────────

def flatten_instances(instance_types: list) -> list[dict]:
    """Flatten nested instance JSON into one row per (instance, region)."""
    today = date.today().isoformat()
    rows = []

    for inst in instance_types:
        config = inst.get("configuration", {})
        price_cents = inst.get("hourly_price", 0)
        price_usd = price_cents / 100.0

        base = {
            "date": today,
            "provider": inst.get("cloud", ""),
            "gpu_model": config.get("gpu_type", ""),
            "num_gpus": config.get("num_gpus", 0),
            "vram_per_gpu_gb": config.get("vram_per_gpu_in_gb", ""),
            "price_per_hour_usd": round(price_usd, 4),
            "interconnect": config.get("interconnect", ""),
        }

        availability = inst.get("availability", [])
        if not availability:
            rows.append({**base, "region": "", "available": False})
            continue

        for avail in availability:
            rows.append({
                **base,
                "region": avail.get("display_name", avail.get("region", "")),
                "available": avail.get("available", False),
            })

    return rows


# ─── Save ────────────────────────────────────────────────────────────

def save_to_csv(records: list[dict], output_path: str) -> pd.DataFrame:
    """Save flattened records to CSV, sorted by gpu_model then price."""
    df = pd.DataFrame(records)

    col_order = [
        "date", "provider", "region", "gpu_model", "num_gpus",
        "vram_per_gpu_gb", "price_per_hour_usd", "available", "interconnect",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["gpu_model", "price_per_hour_usd"], ignore_index=True)

    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df


# ─── Main ────────────────────────────────────────────────────────────

async def main(output_path: str):
    async with aiohttp.ClientSession() as session:
        instance_types = await fetch_instance_types(session)

    if not instance_types:
        logger.warning("No instance data retrieved. Exiting.")
        return

    records = flatten_instances(instance_types)
    df = save_to_csv(records, output_path)

    # Summary stats
    logger.info(
        "Summary: %d rows | %d providers | %d GPU models | price range $%.2f–$%.2f/hr",
        len(df),
        df["provider"].nunique(),
        df["gpu_model"].nunique(),
        df["price_per_hour_usd"].min(),
        df["price_per_hour_usd"].max(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Shadeform GPU rental prices")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output))
