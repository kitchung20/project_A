# project_A

# Detecting Anomalies in Satellite Orbit Data

This is a Data Science Research Project for the Master of Data Science program at the University of Adelaide. The goal of the project is to detect orbital anomalies (such as satellite manoeuvres) using time-series forecasting and residual-based anomaly detection.

## Project Overview

We apply a residual-based anomaly detection framework to the Brouwer Mean Motion (BMM) time series data from three Earth observation satellites:

- **CryoSat-2**
- **SARAL**
- **Sentinel-3A**

Three models were used for forecasting normal BMM behavior:

- **ARIMA** (Autoregressive Integrated Moving Average)
- **XGBoost** (Gradient Boosted Trees)
- **LSTM** (Long Short-Term Memory neural network)

Anomalies are flagged when the model's prediction residual exceeds a threshold selected via F1-score maximization. Detected anomalies are compared with known manoeuvre dates using a ±5-day matching window.

## Project Structure
project_A/
├── data/ # Raw satellite orbital data (TLE)
├── models/ # Model code(ARIMA/XGBoost/LSTM Structures)
├── results/ # Output metrics and figures
├── data_loader.py # Loads BMM and Ground Truth data
├── features.py # Lag feature engineering
├── evaluation.py # Evaluation metrics (Precision, Recall, F1)
├── visualization.py # PR curve and residual plots
├── ground_truth.py # Known manoeuvre events loader
├── main_cs2.py # Run pipeline for CryoSat-2
├── main_s3a.py # Run pipeline for Sentinel-3A
├── main_SARAL.py # Run pipeline for SARAL
├── print_best_metrics_multisat.py # Print best thresholds and metrics
├── all_result.png # Summary figure of all results


## Requirements

- Python 3.10

## How to Run

Clone the repository:
git clone https://github.com/kitchung20/project_A.git
cd project_A

Run one of the main scripts for each satellite:
python main_cs2.py     # For CryoSat-2
python main_SARAL.py   # For SARAL
python main_s3a.py     # For Sentinel-3A

Each script will:
Load BMM and ground truth data
Create lag features and train ARIMA, XGBoost, and LSTM
Predict BMM, calculate residuals, and detect anomalies
Evaluate detection performance using PR metrics

Visualizations and metrics will be printed or saved to /results.


## Methodology Summary
BMM is the primary orbital feature used due to its sensitivity to altitude and manoeuvres.
Residuals (actual - predicted) are calculated for each model.
A fixed threshold (per model and satellite) is selected via validation to maximize the F1 score.
Anomalies are matched to known events within a ±5-day window to compute metrics.

## Acknowledgements
This project was supervised by Dr. David Shorten. We gratefully acknowledge his support and feedback throughout the research.
