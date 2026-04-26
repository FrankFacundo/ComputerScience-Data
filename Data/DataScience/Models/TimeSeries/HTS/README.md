# Hierarchical Time Series Forecasting

This example trains a bottom-up hierarchical forecaster on the public Australian
tourism dataset from Rdatasets:

```text
Australia total -> State -> Region
```

The model can be a global GRU or a small Transformer encoder trained on
bottom-level region series. Inference forecasts every region, then reconciles
state and national forecasts by summing the bottom-level predictions.

## Train

```bash
pip install -r requirements.txt
python train.py --epochs 50 --lookback 12 --horizon 4
```

Train the Transformer version:

```bash
python train.py --model-type transformer --epochs 50 --lookback 12 --horizon 4
```

The script downloads `tourism.csv` into `./data` unless `--data-csv` is passed.
It saves:

- `tourism_hts_checkpoint.pt`
- `tourism_hts_metrics.json`

For a fast smoke test:

```bash
python train.py --epochs 2 --max-series 8 --hidden-dim 32 --batch-size 64
```

## Optuna Comparison

Tune both architectures and compare their validation/test performance:

```bash
python optuna_compare.py --n-trials 12 --trial-epochs 15 --final-epochs 50
```

For a fast smoke test:

```bash
python optuna_compare.py --n-trials 1 --trial-epochs 1 --final-epochs 1 --max-series 4
```

The comparison script writes:

- `tourism_hts_model_comparison.json`
- `tourism_hts_model_comparison.csv`
- `optuna_checkpoints/tourism_hts_gru_best.pt`
- `optuna_checkpoints/tourism_hts_transformer_best.pt`

## Inference

```bash
python inference.py --checkpoint tourism_hts_checkpoint.pt
```

The output CSV contains coherent forecasts for total, state, and region levels:

- `forecast_origin`
- `forecast_quarter`
- `horizon_step`
- `level`
- `state`
- `region`
- `series_id`
- `forecast_trips_thousands`
