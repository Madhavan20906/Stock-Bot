"""
Full pipeline runner: features → LSTM → RL → backtest.
"""

import argparse
import subprocess
import sys
from loguru import logger


def run(name, cmd):
    logger.info(f"\n{'='*50}\n▶  {name}\n{'='*50}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error(f"Step '{name}' failed.")
        sys.exit(1)
    logger.success(f"✓ {name} complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training (slow)")
    args = parser.parse_args()

    run("Train LSTM", ["python", "src/models/lstm/train.py",
                       "--config", "configs/lstm.yaml", "--ticker", args.ticker])

    if not args.skip_rl:
        run("Train RL Agent", ["python", "src/models/rl_agent/train.py",
                                "--config", "configs/rl_agent.yaml", "--ticker", args.ticker])

    run("Run Backtest", ["python", "src/backtesting/engine.py",
                         "--config", args.config])

    run("Unit Tests", ["pytest", "tests/unit/", "--tb=short", "-q"])

    logger.success("Pipeline complete!")


if __name__ == "__main__":
    main()
