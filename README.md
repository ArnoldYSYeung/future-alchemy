# Future Alchemy
Predicting the price of gold 7 days into the future

## Summary
The goal of this project is to predict the price of gold (GC=F) 7 days advance, using technical indicators available, including those from the USD strength index and the price of Bitcoin (virtual gold).

This project contains a machine learning training and evaluation pipeline `Gold Price (GC=F) Prediction.ipynb` and a web application for running and analysing the deployed machine learning models `ui.py`.

## Setup
The following packages are required to run the Jupyter notebook:
- Scikit-Learn 0.22.2 (files in `./models/` are saved with this version)
- YFinance
- TA
- Matplotlib
- Numpy
- Pandas

To run `ui.py` for the web application, <a href="https://www.streamlit.io/">Streamlit</a> is required.

## Run
The machine learning training and evaluation pipeline is in `Gold Price (GC=F) Prediction.ipynb`. This pipeline file will generate DataFrame files for the web application.

To run the web application `ui.py`, after installing Streamlit, go to the main directory and run `streamlight run ui.py`.  This will open the web application in your `localhost`.
