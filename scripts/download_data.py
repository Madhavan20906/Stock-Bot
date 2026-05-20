"""
Download the Kaggle Stock Market Dataset.

Requires:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY in your .env or ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KAGGLE_DATASET = "jacksoncrow/stock-market-dataset"
OUTPUT_PATH = Path("data/raw")


def check_kaggle_creds():
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print("ERROR: Kaggle credentials not found.")
            print("Either:")
            print("  1. Set KAGGLE_USERNAME and KAGGLE_KEY in .env")
            print("  2. Place kaggle.json in ~/.kaggle/kaggle.json")
            print("\nGet your API key at: https://www.kaggle.com/account")
            sys.exit(1)
    else:
        # Write to ~/.kaggle/kaggle.json for the CLI
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        import json
        with open(kaggle_dir / "kaggle.json", "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_dir / "kaggle.json", 0o600)


def download():
    check_kaggle_creds()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{KAGGLE_DATASET}' to {OUTPUT_PATH}...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(OUTPUT_PATH), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Download failed:\n{result.stderr}")
        sys.exit(1)

    stocks = list(OUTPUT_PATH.glob("stocks/*.csv"))
    etfs = list(OUTPUT_PATH.glob("etfs/*.csv"))
    print(f"\nDownload complete!")
    print(f"  Stocks: {len(stocks)} files")
    print(f"  ETFs:   {len(etfs)} files")
    print(f"\nYou're ready to run: python scripts/run_pipeline.py")


if __name__ == "__main__":
    download()
