"""Unit tests for the StockTradingEnv Gymnasium environment."""

import numpy as np
import pytest
from src.models.rl_agent.environment import StockTradingEnv


@pytest.fixture
def env():
    n = 200
    features = np.random.randn(n, 10).astype(np.float32)
    prices = np.abs(np.cumsum(np.random.randn(n))) + 100.0
    return StockTradingEnv(
        df_features=features,
        prices=prices,
        initial_balance=10_000,
        transaction_cost=0.001,
        window_size=20,
    )


def test_reset_returns_correct_shape(env):
    obs, info = env.reset()
    expected = env.observation_space.shape[0]
    assert obs.shape == (expected,), f"Expected obs shape ({expected},), got {obs.shape}"


def test_observation_space_bounds(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs.astype(np.float32))


def test_buy_action_reduces_cash(env):
    env.reset()
    cash_before = env.cash
    env.step(np.array([1.0]))  # Full buy
    assert env.cash < cash_before


def test_sell_action_reduces_shares(env):
    env.reset()
    env.step(np.array([1.0]))  # Buy first
    shares_after_buy = env.shares_held
    env.step(np.array([-1.0]))  # Full sell
    assert env.shares_held < shares_after_buy


def test_hold_action_no_change(env):
    env.reset()
    env.step(np.array([0.5]))   # Buy some
    cash_before = env.cash
    shares_before = env.shares_held
    env.step(np.array([0.0]))   # Hold
    assert env.cash == pytest.approx(cash_before)
    assert env.shares_held == pytest.approx(shares_before)


def test_episode_terminates(env):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert steps < 10_000, "Episode never terminated"
    assert done


def test_portfolio_value_never_negative(env):
    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        assert info["portfolio_value"] >= 0
        if terminated or truncated:
            break


def test_transaction_cost_applied(env):
    env.reset()
    env.shares_held = 0.0
    cash_before = env.cash
    env.step(np.array([1.0]))
    # With 0.1% cost, shares × price should be slightly less than cash spent
    price = env.prices[env.current_step - 1]
    assert env.shares_held * price < cash_before * (1 - 1e-6)
