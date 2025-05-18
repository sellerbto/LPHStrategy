## Liquidity-provision plus delta-neutral hedge strategy

### Idea:

Strategy pairs concentrated ETH/USDT liquidity on Uniswap V3 with a hedge on
Hyperliquid perps, neutralizing ETH exposure.

### [Paper](LPHStrategy.pdf)


### Start backtest

```shell
uv venv -p 3.12
```

```shell
uv run ./liquidity_provider_hedge/lph_pipeline.py
```