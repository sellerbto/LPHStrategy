from dataclasses import dataclass
from typing import List

from fractal.core.entities import HyperliquidEntity, UniswapV3LPEntity, UniswapV3LPConfig, UniswapV3LPGlobalState, HyperLiquidGlobalState
from fractal.core.entities.uniswap_v3_lp import UniswapV3LPInternalState
from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, EthereumUniswapV3Loader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory, RateHistory
from fractal.loaders.hyperliquid import HyperliquidFundingRatesLoader

from datetime import datetime, UTC
import pandas as pd
from collections import namedtuple
from dotenv import load_dotenv
import os

load_dotenv()

LPPrices = namedtuple('LPPrices', ['pl', 'p', 'ph'])

THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')

@dataclass
class LPHParams(BaseStrategyParams):
    """
    Hyperparameters for LP + Hedge on HyperLiquid strategy:
    - TAU: width of Uniswap V3 range in ticks
    - INITIAL_BALANCE: total notional to deploy
    - HEDGE_REBALANCE_THRESHOLD: min change in ETH exposure to rebalance hedge
    """
    TAU: float
    INITIAL_BALANCE: float
    HEDGE_REBALANCE_THRESHOLD: float


# noinspection PyTypeChecker
class LPHStrategy(BaseStrategy):
    # Pool-specific constants (set before use)
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: LPHParams, debug: bool = False, *args, **kwargs):
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial = False

    def set_up(self):
        self.register_entity(NamedEntity('HEDGE', HyperliquidEntity()))
        self.register_entity(NamedEntity('UNISWAP_V3',
            UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
                )
            )
        ))
        assert isinstance(self.get_entity('HEDGE'), HyperliquidEntity)
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def predict(self) -> List[ActionToTake]:
        hedge: HyperliquidEntity = self.get_entity('HEDGE')
        lp: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        params: LPHParams = self._params

        pl, p, ph = lp.internal_state.price_lower, lp.global_state.price, lp.internal_state.price_upper

        actions: List[ActionToTake] = []

        if not self.deposited_initial and hedge.balance == 0 and lp.balance == 0:
            self._debug("Initial deposit required - no positions open")
            self.deposited_initial = True
            return self._deposit_into_strategy()

        lp_rebalanced = False
        if p < pl or p > ph:
            self._debug(f"Price {p} out of [{pl},{ph}], rebalance LP")
            lp_rebalanced = True
            actions += self._rebalance_lp()

        delta_lp = self._compute_lp_delta(p, lp_rebalanced)
        desired_hedge = -delta_lp
        current_hedge = hedge.size

        if abs(desired_hedge) < 1e-6: 
            hedge_deviation = abs(current_hedge) 
        else:
            hedge_deviation = abs(current_hedge - desired_hedge) / abs(desired_hedge)
        if hedge_deviation > params.HEDGE_REBALANCE_THRESHOLD:
            diff = desired_hedge - current_hedge
            self._debug(f"Rebalancing hedge: current {current_hedge:.4f}, target {desired_hedge:.4f}, diff {diff:.4f}")
            actions += self._rebalance_hedge(diff)

        return actions
       
    def _rebalance_hedge(self, diff) -> List[ActionToTake]:
        self._debug(f"{'Increasing' if diff > 0 else 'Decreasing'} hedge position by {diff:.4f}")

        return [
            ActionToTake('HEDGE', Action('open_position', {'amount_in_product': diff})),
            
        ]

    def _rebalance_lp(self) -> List[ActionToTake]:
        lp: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        actions: List[ActionToTake] = []

        lp_prices = self._get_lp_prices()
        pl, ph = lp_prices.pl, lp_prices.ph
        
        if lp.internal_state.liquidity > 0:
            actions.append(ActionToTake('UNISWAP_V3', Action('close_position', {})))
            self._debug("Liquidity withdrawn from the current range.")
        
        cash_lambda = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake('UNISWAP_V3', Action('open_position', {
            'amount_in_notional': cash_lambda, # Allocate all available cash
            'price_lower': pl,
            'price_upper': ph,
        })))
        self._debug(f'New position opened with range [{pl}, {ph}]') 

        return actions

    def _deposit_into_strategy(self) -> List[ActionToTake]:
        """
        Deposit initial balance and open a position.
        """
        params: LPHParams = self._params

        lp_prices = self._get_lp_prices()
        pl, p, ph = lp_prices.pl, lp_prices.p, lp_prices.ph
        
        lp_cap, hedge_cap = LPHStrategy._compute_initial_allocation(params.INITIAL_BALANCE, p, pl, ph)
        
        self._debug(f"Initial deposit: {params.INITIAL_BALANCE:.2f}")
        self._debug(f"LP allocation: {lp_cap:.2f}, Hedge allocation: {hedge_cap:.2f}")
        self._debug(f"Initial LP range: [{pl:.2f}, {ph:.2f}]")

        product_to_hedge_lambda = lambda obj: -self._compute_lp_delta(p, False)
        
        return [
            ActionToTake('UNISWAP_V3', Action('deposit', {'amount_in_notional': lp_cap})),
            ActionToTake('HEDGE', Action('deposit', {'amount_in_notional': hedge_cap})),
            ActionToTake('UNISWAP_V3', Action('open_position', {
                'amount_in_notional': lp_cap,
                'price_lower': pl,
                'price_upper': ph
            })),
            ActionToTake('HEDGE', Action('open_position', {
                'amount_in_product': product_to_hedge_lambda
            })),
        ]
    
    def _get_lp_prices(self) -> LPPrices:
        params: LPHParams = self._params
        lp: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        lp_global_state: UniswapV3LPGlobalState = lp.global_state

        p = lp_global_state.price

        pl = p * 1.0001 ** (-params.TAU * self.tick_spacing)
        ph = p * 1.0001 ** (params.TAU * self.tick_spacing)

        return LPPrices(pl=pl, p=p, ph=ph)

    def _compute_lp_delta(self, p: float, compute_pl_ph: bool) -> float:
        lp_internal_state: UniswapV3LPInternalState = self.get_entity('UNISWAP_V3').internal_state
        L = lp_internal_state.liquidity

        if compute_pl_ph:
            lp_prices = self._get_lp_prices()
            pl, ph = lp_prices.pl, lp_prices.ph
        else:
            pl, ph = lp_internal_state.price_lower, lp_internal_state.price_upper

        if p < pl:
            return L * (1/pl**0.5 - 1/ph**0.5)
        if p > ph:
            return 0.0
        return L * (1/p**0.5 - 1/ph**0.5)
    
    @staticmethod
    def _compute_initial_allocation(initial_balance, p, pl, ph):
        # in our case it sensless to count delta up, delta down but in the future...
        delta_up   = (ph - p) / p
        delta_down = (p - pl) / p
        delta_adv  = max(delta_up, delta_down)

        L_max = 1.0 / (delta_adv)

        # i want maximum safety, so I make hedge allocation in the way it never liquidates
        s = 0.8
        L_target = s * L_max

        hedge_cap = initial_balance / (1 + L_target)
        lp_cap    = initial_balance - hedge_cap

        return lp_cap, hedge_cap

    

def get_observations(
        rate_date: RateHistory, pool_data: PoolHistory, binance_prices: PriceHistory,
        start_time: datetime = None, end_time: datetime = None
) -> List[Observation]:
    observations_df: pd.DataFrame = pool_data.join(binance_prices).join(rate_date)
    observations_df['rate'] = observations_df['rate'].fillna(0)
    observations_df = observations_df.loc[start_time:end_time]
    observations_df = observations_df.dropna()
    observations_df = observations_df.sort_index()
    return [
        Observation(
            timestamp=timestamp,
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(price=price, tvl=tvl, volume=volume, fees=fees, liquidity=liquidity),
                'HEDGE': HyperLiquidGlobalState(mark_price=price, funding_rate=rate),

            }
        ) for timestamp, (tvl, volume, fees, liquidity, price, rate) in observations_df.iterrows()
    ]



def build_observations(
    binance_ticker: str, hyperliquid_ticker: str, pool_address: str, api_key: str,
    start_time: datetime = None, end_time: datetime = None, fidelity: str = 'hour' 
) -> List[Observation]:
    if fidelity == 'hour':
        rate_data: RateHistory = HyperliquidFundingRatesLoader(
            hyperliquid_ticker, start_time=start_time, end_time=end_time).read(with_run=True)
        
        pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
                api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        
        binance_prices: PriceHistory = BinanceHourPriceLoader(binance_ticker, loader_type=LoaderType.CSV, start_time=start_time, end_time=end_time).read(with_run=True)
    if fidelity == 'minute':
        rate_data: RateHistory = HyperliquidFundingRatesLoader(
            hyperliquid_ticker, start_time=start_time, end_time=end_time).read(with_run=True)
        rate_data = rate_data[~rate_data.index.duplicated(keep='first')]
        rate_data = rate_data.resample('1min').ffill()
        
        pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
                api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        
        binance_prices: PriceHistory = BinanceMinutePriceLoader(binance_ticker, loader_type=LoaderType.CSV, start_time=start_time, end_time=end_time).read(with_run=True)
        
    return get_observations(rate_data, pool_data, binance_prices, start_time, end_time)


if __name__ == '__main__':
    binance_ticker: str = 'ETHUSDT'
    hyperliquid_ticker: str = 'ETH'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'

    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)
    
    params = LPHParams(TAU=15, INITIAL_BALANCE=1_000_000, HEDGE_REBALANCE_THRESHOLD=0.02)
    LPHStrategy.token0_decimals = token0_decimals
    LPHStrategy.token1_decimals = token1_decimals
    LPHStrategy.tick_spacing = 60

    strategy: LPHStrategy = LPHStrategy(debug=True, params=params)

    entities = strategy.get_all_available_entities().keys()
    observations: List[Observation] = build_observations(
        binance_ticker=binance_ticker, hyperliquid_ticker=hyperliquid_ticker, pool_address=pool_address, api_key=THE_GRAPH_API_KEY,
        start_time=datetime(2023, 1, 1, tzinfo=UTC), end_time=datetime(2025, 1, 1, tzinfo=UTC), fidelity='hour'
    )
    observation0 = observations[0]
    assert all(entity in observation0.states for entity in entities)

    result = strategy.run(observations)
    print(result.get_default_metrics())
    result.to_dataframe().to_csv('lph_strategy_result.csv')  
    print(result.to_dataframe().iloc[-1]) 