from setuptools import setup, find_packages

setup(
    name="stockbot",
    version="1.0.0",
    description="AI Stock Market Prediction & Trading System",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "stockbot-train=models.lstm.train:main",
            "stockbot-backtest=backtesting.engine:main",
            "stockbot-api=api.main:start",
        ]
    },
)
