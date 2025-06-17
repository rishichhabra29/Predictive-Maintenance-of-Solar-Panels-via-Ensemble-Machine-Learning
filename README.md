# Predictive Maintenance of Solar Panels via Ensemble Machine Learning

## About
Built for the Zelestra × AWS ML Ascend Challenge 2nd Edition. End-to-end ML pipeline forecasting solar panel efficiency from sensor data. Features: target-encoded IDs, error-severity, interaction/polynomial/cyclical/rolling stats. Optuna-tuned XGBoost/LightGBM/CatBoost with early stopping, stacked ensemble — 89.81% score, ranked 148/1016.

## Installation
```bash
git clone https://github.com/yourusername/solar-pv-predictive-maintenance-ensemble-ml.git
cd solar-pv-predictive-maintenance-ensemble-ml
pip install -r requirements.txt

Usage
Place train.csv and test.csv in the repository root.

Run training & inference:
python train_model2.py
A submission.csv file will be produced.

Repository Structure
├── feature_engineering.py   # data cleaning & feature creation
├── train_model2.py          # Optuna-tuned, early-stopped stacking pipeline
├── requirements.txt         # Python dependencies
├── train.csv                # training data (not committed)
├── test.csv                 # test data (not committed)
└── submission.csv           # generated output
Results
Validation Score: 89.81% (100×(1−RMSE))

Challenge Rank: 148 out of 1016 teams

By OSM_Knights
