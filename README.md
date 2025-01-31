## Time Series Analysis with ARIMA and LSTM

This repository contains a Python script for performing time series analysis using ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory) models. The script includes data preprocessing, decomposition, stationarity checks, and forecasting.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Time series analysis is a statistical technique that deals with time series data, or data that is observed sequentially over time. This project demonstrates how to analyze and forecast time series data using ARIMA and LSTM models. The script includes steps for data visualization, decomposition, stationarity checks, and model evaluation.

## Features

- **Data Visualization**: Plot the original time series data.
- **Decomposition**: Decompose the time series into trend, seasonal, and residual components.
- **Stationarity Check**: Use the Augmented Dickey-Fuller test to check for stationarity.
- **ARIMA Model**: Fit an ARIMA model to the time series data and make predictions.
- **LSTM Model**: Build and train an LSTM model for time series forecasting.
- **Model Evaluation**: Evaluate the models using RMSE (Root Mean Squared Error) and visualize the actual vs. predicted values.

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- tensorflow

You can install these dependencies using the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/time-series-analysis.git
