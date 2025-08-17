# Advanced Time Series Forecasting for Stock Prices using Deep Learning

As a data scientist specializing in AI and deep learning, I've developed this project to demonstrate my hands-on experience in building and comparing various time series forecasting models for financial data. Using Microsoft (MSFT) stock prices as a case study, this notebook showcases an end-to-end workflow, including data acquisition, environment setup for GPU-accelerated training, data preprocessing, and the implementation of multiple forecasting architectures.

This work highlights my ability to leverage a range of powerful libraries—from foundational ones like **TensorFlow/Keras** to advanced frameworks like **GluonTS** and **Darts**. It demonstrates not just the implementation of models but also an understanding of their underlying principles, from simple RNNs to probabilistic forecasting with DeepAR. The inclusion of different techniques is intentional, showcasing my skills in benchmarking, practical application, and continuous learning.

-----

## Problem Statement and Goal of Project

Financial markets generate vast amounts of time series data, and accurately forecasting stock prices is crucial for informing investment strategies and managing risk. The goal of this project is to explore deep learning for this task by:

1.  Retrieving historical Microsoft (MSFT) stock data from **2012 to 2021**.
2.  Preprocessing the data for sequential modeling.
3.  Implementing and training several time series models: **Simple RNN**, **LSTM**, **DeepAR (via GluonTS)**, and an **RNN model (via Darts)**.
4.  Visualizing and comparing the forecasts to illustrate the strengths of different approaches, particularly the value of probabilistic forecasting for capturing uncertainty.

This project serves as a practical, in-depth exploration of deep learning techniques for sequential data, emphasizing data handling, model implementation, and comparative analysis.

-----

## Solution Approach

The approach is structured to build from foundational models to more advanced, industry-standard frameworks.

  - **Environment Verification**: The notebook begins by checking **TensorFlow (v2.10.0)** and **PyTorch (v2.5.1)** versions and confirming GPU availability (1 GPU detected for both) to ensure accelerated computations.
  - **Data Acquisition & Preprocessing**: Historical MSFT stock data is downloaded using the `yfinance` library. The data is loaded into a Pandas DataFrame, and key price columns (`Adj Close`, `Open`, `Close`, `High`, `Low`) are selected and prepared for modeling. A custom function creates training and testing sequences from the time series.
  - **Baseline Models (Keras)**:
      - **Simple RNN**: A baseline `Sequential` model with two `SimpleRNN` layers is built to establish a performance benchmark.
      - **LSTM**: An `LSTM` model with a similar architecture is implemented to demonstrate its advantages in capturing long-term dependencies over a Simple RNN.
  - **Probabilistic Forecasting (GluonTS & DeepAR)**:
      - To move beyond single-point predictions, Amazon's **DeepAR** model is implemented using the **GluonTS** library. This powerful autoregressive model learns a probabilistic distribution of future values, providing not just a forecast but also confidence intervals (quantile bands), which are critical for risk assessment in finance.
  - **Historical Forecasting (Darts)**:
      - The **Darts** library is used to implement another RNN model, showcasing its user-friendly API for time series manipulation. The notebook demonstrates how to generate **historical forecasts**, which involves training the model on an expanding window of historical data to simulate its performance over time. This backtesting approach is crucial for validating a model's real-world applicability.
  - **Visualization**: Custom plotting functions are used to visualize the forecasts from each model against the actual stock prices, including quantile bands for the probabilistic DeepAR model.

-----

## Technologies & Libraries

  - **Programming Language**: Python (via Jupyter Notebook)
  - **Deep Learning Frameworks**: TensorFlow (v2.10.0), Keras, PyTorch (v2.5.1)
  - **Time Series Libraries**: GluonTS (for DeepAR), Darts (for RNN modeling & backtesting)
  - **Data Processing & Visualization**: NumPy, Pandas, Matplotlib, scikit-learn (`MinMaxScaler`)
  - **Data Retrieval**: `yfinance`, `requests`

-----

## Description about Dataset

The dataset is historical stock price data for **Microsoft (MSFT)**, fetched dynamically via `yfinance` for the period of **January 1, 2012, to December 31, 2021**.

  - **Features**: `Adj Close`, `Close`, `High`, `Low`, `Open`, `Volume`. The `Close` price is used as the target variable for forecasting.
  - **Data Size**: 2516 rows (trading days).
  - **Structure**: The data is a time-indexed Pandas DataFrame, split into a training set (`2012-2020`) and a test set (`2021`).

A sample of the preprocessed data is shown below:

| Date | Adj Close | Open | Close | High | Low |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2012-01-03| 21.039 | 26.55 | 26.77 | 26.96 | 26.39 |
| 2012-01-04| 21.534 | 26.82 | 27.40 | 27.47 | 26.78 |
| 2012-01-05| 21.754 | 27.38 | 27.68 | 27.73 | 27.29 |
| 2012-01-06| 22.092 | 27.53 | 28.11 | 28.19 | 27.53 |
| 2012-01-09| 21.802 | 28.05 | 27.74 | 28.10 | 27.72 |

-----

## Installation & Execution Guide

1.  **Prerequisites**:

      - Python 3.9+
      - Jupyter Notebook or JupyterLab
      - GPU hardware is recommended for faster training.

2.  **Install Dependencies**:

    ```bash
    pip install tensorflow==2.10.0 torch==2.5.1 numpy pandas matplotlib scikit-learn yfinance requests keras gluonts "darts[torch]"
    ```

3.  **Execution**:

      - Open the `Deep_ar_me.ipynb` notebook in your Jupyter environment.
      - Run the cells sequentially to perform environment checks, download data, preprocess it, and train each of the models (RNN, LSTM, DeepAR, and Darts RNN).
      - The notebook handles SSL verification issues with `yfinance` for seamless data retrieval.

-----

## Key Results / Performance

  - **Environment Setup**: Successfully detected both TensorFlow (`PhysicalDevice:GPU:0`) and PyTorch CUDA (`True`, device count: 1), enabling efficient GPU-accelerated training.
  - **Data Retrieval**: Downloaded 2516 rows of MSFT data without errors.
  - **Model Training**: All four models (Simple RNN, LSTM, DeepAR, and Darts RNN) were successfully trained on the historical data. The Keras models demonstrated rapid convergence, while GluonTS and Darts showcased robust training loops for more complex forecasting tasks.
  - **Forecasting & Visualization**:
      - The Simple RNN and LSTM models produced coherent point forecasts, with the LSTM showing a slightly better fit to the validation data.
      - The **DeepAR** model successfully generated probabilistic forecasts, providing a median prediction along with 50% and 90% confidence intervals, which effectively captured the price volatility in 2021.
      - The plot outputs clearly visualize the predicted price movements against the actual values, offering a direct comparison of each model's performance.

-----

## Screenshots / Sample Outputs

  - **TensorFlow & PyTorch GPU Verification**:

    ```
    TensorFlow version: 2.10.0
    Detected GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    torch version: 2.5.1
    True
    1
    ```

  - **Raw Data Sample from `yfinance`**:

    ```
    Price         Adj Close       Close        High         Low        Open    Volume
    Ticker             MSFT        MSFT        MSFT        MSFT        MSFT      MSFT
    Date
    2012-01-03    21.039213   26.770000   26.959999   26.389999   26.549999  64731500
    ...                 ...         ...         ...         ...         ...       ...
    2021-12-30   329.475433  339.320007  343.130005  338.820007  341.910004  15994500
    ```

  - **Plot Outputs**: The notebook generates multiple Matplotlib figures visualizing the forecasts from each model, including one with quantile bands from the DeepAR model, which is essential for understanding prediction uncertainty.

-----

## Additional Learnings / Reflections

This project provided deep, practical insights into the nuances of time series forecasting with deep learning.

  - **Framework Comparison**: Implementing models in Keras, GluonTS, and Darts highlighted the trade-offs between ease of use and advanced functionality. While Keras is excellent for rapid prototyping, GluonTS and Darts offer specialized, powerful tools for probabilistic forecasting and backtesting that are indispensable for production-level financial modeling.
  - **Probabilistic vs. Point Forecasts**: The output from the DeepAR model underscored the importance of probabilistic forecasting. For financial applications, understanding the range of possible outcomes (uncertainty) is often more valuable than a single point estimate.
  - **Practical Challenges**: I gained experience in handling common data science challenges, such as API quirks (`yfinance` SSL issues) and ensuring cross-framework compatibility (TensorFlow vs. PyTorch GPU setup).

This end-to-end project serves as a strong foundation and benchmark, paving the way for exploring even more advanced architectures like Transformers for time series in production scenarios.

-----

## 👤 Author

## Mehran Asgari

**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*