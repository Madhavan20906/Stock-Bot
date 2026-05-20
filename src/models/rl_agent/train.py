"""
Train a PPO agent on the StockTradingEnv using stable-baselines3.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.data.loader import load_ticker
from src.data.preprocessor import StockPreprocessor
from src.features.technical import add_all_indicators
from src.models.rl_agent.environment import StockTradingEnv


ALGO_MAP = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


def make_env(ticker: str, config: dict, split: str = "train"):
    """Build a StockTradingEnv for the given ticker and split."""
    df = load_ticker(ticker)
    df = add_all_indicators(df, config.get("features", {}).get("technical", {}))

    prep = StockPreprocessor(sequence_length=1, target_col="close")
    prep.fit(df)
    scaled = prep.transform(df)

    data_cfg = config["data"]
    if split == "train":
        mask = df.index <= data_cfg["train_end"]
    elif split == "val":
        mask = (df.index > data_cfg["train_end"]) & (df.index <= data_cfg["val_end"])
    else:
        mask = df.index > data_cfg["val_end"]

    sub_scaled = scaled[mask]
    sub_prices = df.loc[mask, "close"].values

    env_cfg = config["environment"]
    return StockTradingEnv(
        df_features=sub_scaled.values,
        prices=sub_prices,
        initial_balance=env_cfg["initial_balance"],
        transaction_cost=env_cfg["transaction_cost"],
        window_size=env_cfg["window_size"],
        reward_scaling=env_cfg["reward_scaling"],
    )


def train(ticker: str, config: dict):
    agent_cfg = config["agent"]
    train_cfg = config["training"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(ticker, config, "train")
    eval_env = make_env(ticker, config, "val")

    AlgoClass = ALGO_MAP.get(agent_cfg["algorithm"], PPO)

    model = AlgoClass(
        policy=agent_cfg["policy"],
        env=train_env,
        n_steps=agent_cfg["n_steps"],
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        learning_rate=agent_cfg["learning_rate"],
        verbose=agent_cfg["verbose"],
        tensorboard_log=train_cfg.get("tensorboard_log"),
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(checkpoint_dir),
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
        callback_after_eval=stop_callback,
        verbose=1,
    )

    logger.info(f"Training {agent_cfg['algorithm']} agent on {ticker} for {train_cfg['total_timesteps']:,} steps")
    model.learn(total_timesteps=train_cfg["total_timesteps"], callback=eval_callback)
    model.save(checkpoint_dir / f"final_{ticker}")
    logger.info(f"Agent saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_agent.yaml")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(args.ticker, config)


if __name__ == "__main__":
    main()
