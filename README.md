# Stock Market Prediction using Economic Indicators

## Project Overview
End-to-end machine learning pipeline to predict the closing price of the Dow Jones Industrial Average (DJI)
using technical indicators, macroeconomic variables, commodities, and global indices.

## Features
- Modular ML architecture
- Config-driven pipeline
- Time-series aware split
- XGBoost / RandomForest support
- Docker-ready
- API-ready (FastAPI)

## Run Locally
1. Install dependencies:
   pip install -r requirements.txt

2. Run training:
   python main.py

## Docker
docker build -t stock-prediction .
docker run stock-prediction