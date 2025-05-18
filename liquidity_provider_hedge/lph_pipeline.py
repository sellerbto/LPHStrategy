import os
from typing import List
import warnings

import numpy as np
from datetime import datetime, UTC
from dotenv import load_dotenv
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from lph_strategy import LPHStrategy, build_observations, Observation, THE_GRAPH_API_KEY, EthereumUniswapV3Loader, LoaderType

warnings.filterwarnings('ignore')
load_dotenv()

def build_grid():
    return ParameterGrid({
        'TAU': np.linspace(start=3, stop=30, num=15, dtype=int),
        'INITIAL_BALANCE': [1_000_000],
        'HEDGE_REBALANCE_THRESHOLD': np.linspace(0.02, 0.5, 9),
    })

if __name__ == '__main__':
    binance_ticker: str = 'ETHUSDT'
    hyperliquid_ticker: str = 'ETH'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'

    start_time = datetime(2022, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    fidelity = 'hour'
    experiment_name = f'lph_{fidelity}_{binance_ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}_month_window_huge_tau_2'

    mlflow_uri = os.getenv('MLFLOW_URI')
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)

    LPHStrategy.token0_decimals = token0_decimals
    LPHStrategy.token1_decimals = token1_decimals
    LPHStrategy.tick_spacing = 60

    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    observations: List[Observation] = build_observations(
        binance_ticker=binance_ticker, hyperliquid_ticker=hyperliquid_ticker, pool_address=pool_address, api_key=THE_GRAPH_API_KEY,
        start_time=start_time, end_time=end_time, fidelity=fidelity
    )
    assert len(observations) > 0

    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=LPHStrategy,
        backtest_observations=observations,
        window_size=24 * 30,
        params_grid=build_grid(),
        debug=True,
    )    

    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()

    