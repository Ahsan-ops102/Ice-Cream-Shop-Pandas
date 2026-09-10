# Ice Cream Sales Data-Preparation Exercises

A collection of small Pandas exercises that demonstrate time-series leakage, missing-value handling, rolling features, and outlier detection. Synthetic ice-cream sales data keeps each concept easy to inspect and reproduce.

## What is included

- Correct and incorrect seven-day moving averages for forecasting.
- Six missing-value strategies: zero, mean, median, forward fill, backward fill, and interpolation.
- A visual explanation of why backward filling can leak future information into a prediction workflow.
- Z-score and interquartile-range outlier detection.
- Histograms, box plots, time-series comparisons, and annotated scatter plots.

## Repository contents

| File | Purpose |
| --- | --- |
| `ice_cream_analysis.py` | Compares leaky and shifted rolling-average features |
| `missing_values_cleanup.py` | Demonstrates six ways to fill missing sales values |
| `visualize_data_leakage.py` | Visualizes forward fill versus future-looking backward fill |
| `outlier_detection.py` | Compares Z-score and IQR outlier rules |
| `data_leakage_comparison.png` | Saved leakage visualization |
| `outlier_detection.png` | Saved outlier visualization |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
python ice_cream_analysis.py
python missing_values_cleanup.py
python visualize_data_leakage.py
python outlier_detection.py
```

The scripts generate their datasets in memory with fixed random seeds where randomness is used. Plotting scripts open Matplotlib windows and overwrite their corresponding PNG outputs in the project directory.
