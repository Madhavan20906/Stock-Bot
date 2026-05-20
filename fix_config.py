import yaml

config = {
    'data': {
        'raw_path': 'data/raw',
        'sequence_length': 60,
        'train_end': '2018-12-31',
        'val_end': '2019-12-31',
    },
    'features': {
        'technical': {
            'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26,
            'macd_signal': 9, 'bb_period': 20, 'bb_std': 2.0, 'atr_period': 14
        }
    },
    'environment': {
        'initial_balance': 100000,
        'transaction_cost': 0.001,
        'max_position_size': 0.2,
        'reward_scaling': 0.0001,
        'window_size': 30,
    },
    'agent': {
        'algorithm': 'PPO',
        'policy': 'MlpPolicy',
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'learning_rate': 0.0003,
        'verbose': 1,
    },
    'training': {
        'total_timesteps': 100000,
        'eval_freq': 10000,
        'n_eval_episodes': 3,
        'checkpoint_dir': 'checkpoints/rl/',
        'tensorboard_log': None,
    },
    'logging': {
        'level': 'INFO',
        'mlflow_tracking_uri': 'mlruns/',
        'experiment_name': 'stockbot_rl'
    }
}

with open('configs/rl_agent.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('rl_agent.yaml written successfully!')