"""
Custom OpenAI Gymnasium environment for stock trading.

Observation space: window of scaled OHLCV + technical features + portfolio state
Action space:      Continuous [-1, 1] per ticker (negative=sell, positive=buy)
Reward:            Sharpe-ratio-adjusted portfolio return
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger


class StockTradingEnv(gym.Env):
    """Single-asset stock trading environment.

    The agent observes the last `window_size` days of features
    plus its current position and cash ratio, and outputs a continuous
    action in [-1, 1] representing what fraction of available cash to
    buy (positive) or what fraction of holdings to sell (negative).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_features: np.ndarray,      # (n_days, n_features) — pre-scaled
        prices: np.ndarray,            # (n_days,) — raw close prices
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 30,
        reward_scaling: float = 1e-4,
    ):
        super().__init__()

        self.features = df_features
        self.prices = prices
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        self.n_features = df_features.shape[1]
        self.n_steps = len(prices) - window_size - 1

        # Observation: (window_size × n_features) + [cash_ratio, position_ratio, unrealized_pnl]
        obs_size = window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: fraction to buy/sell in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_values = [self.initial_balance]
        self.returns_history = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        window = self.features[self.current_step - self.window_size : self.current_step]
        portfolio_value = self._portfolio_value()
        cash_ratio = self.cash / (portfolio_value + 1e-9)
        stock_ratio = (self.shares_held * self.prices[self.current_step]) / (portfolio_value + 1e-9)
        unrealized_pnl = (self.shares_held * self.prices[self.current_step]) / self.initial_balance
        extra = np.array([cash_ratio, stock_ratio, unrealized_pnl], dtype=np.float32)
        return np.concatenate([window.flatten(), extra])

    def _portfolio_value(self) -> float:
        return self.cash + self.shares_held * self.prices[self.current_step]

    def step(self, action: np.ndarray):
        action = float(np.clip(action[0], -1.0, 1.0))
        price = self.prices[self.current_step]
        prev_value = self._portfolio_value()

        if action > 0:  # Buy
            spend = self.cash * action
            shares_to_buy = (spend * (1 - self.transaction_cost)) / price
            self.shares_held += shares_to_buy
            self.cash -= spend

        elif action < 0:  # Sell
            shares_to_sell = self.shares_held * abs(action)
            proceeds = shares_to_sell * price * (1 - self.transaction_cost)
            self.shares_held -= shares_to_sell
            self.cash += proceeds

        self.current_step += 1
        new_value = self._portfolio_value()

        # Reward: log return, scaled
        log_return = np.log(new_value / (prev_value + 1e-9))
        self.returns_history.append(log_return)
        self.portfolio_values.append(new_value)

        # Sharpe adjustment (rolling 20-step)
        if len(self.returns_history) >= 20:
            r = np.array(self.returns_history[-20:])
            sharpe = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
            reward = log_return + 0.1 * sharpe
        else:
            reward = log_return

        reward *= self.reward_scaling

        terminated = self.current_step >= self.window_size + self.n_steps
        truncated = False
        info = {
            "portfolio_value": new_value,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "return": (new_value - self.initial_balance) / self.initial_balance,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        value = self._portfolio_value()
        ret = (value - self.initial_balance) / self.initial_balance * 100
        logger.info(
            f"Step {self.current_step} | Value: ${value:,.0f} | Return: {ret:.2f}% | "
            f"Cash: ${self.cash:,.0f} | Shares: {self.shares_held:.2f}"
        )
