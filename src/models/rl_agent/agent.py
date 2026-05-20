"""
PPO Agent wrapper: inference-time interface for the trained RL agent.
Loads a saved stable-baselines3 checkpoint and exposes a simple predict() API.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from loguru import logger
from stable_baselines3 import PPO, A2C, SAC

from src.models.rl_agent.environment import StockTradingEnv

ALGO_MAP = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


class TradingAgent:
    """Wraps a trained SB3 model for inference."""

    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm
        self.model = None

    def load(self, path: str | Path) -> "TradingAgent":
        """Load a saved checkpoint."""
        path = Path(path)
        if not path.exists():
            # SB3 appends .zip
            path = Path(str(path) + ".zip")
        if not path.exists():
            raise FileNotFoundError(
                f"No RL checkpoint at {path}. "
                "Train the agent first: python src/models/rl_agent/train.py"
            )
        AlgoClass = ALGO_MAP.get(self.algorithm, PPO)
        self.model = AlgoClass.load(str(path))
        logger.info(f"Loaded {self.algorithm} agent from {path}")
        return self

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> float:
        """Return a single action value in [-1, 1].

        Args:
            observation: Flattened observation vector from StockTradingEnv
            deterministic: If True, use greedy policy (recommended for live/backtest)

        Returns:
            action scalar in [-1, 1]
        """
        if self.model is None:
            raise RuntimeError("Agent not loaded. Call .load() first.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return float(np.clip(action[0], -1.0, 1.0))

    def evaluate(
        self,
        env: StockTradingEnv,
        n_episodes: int = 1,
        deterministic: bool = True,
    ) -> dict:
        """Run n_episodes on the given environment and return performance stats.

        Returns:
            Dict with mean_return, std_return, mean_portfolio_value, n_trades_approx
        """
        if self.model is None:
            raise RuntimeError("Agent not loaded. Call .load() first.")

        episode_returns = []
        final_values = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_returns.append(info["return"])
            final_values.append(info["portfolio_value"])
            logger.debug(f"Episode {ep+1}: return={info['return']:.2%}, value=${info['portfolio_value']:,.0f}")

        return {
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_portfolio_value": float(np.mean(final_values)),
            "best_return": float(np.max(episode_returns)),
            "worst_return": float(np.min(episode_returns)),
        }
