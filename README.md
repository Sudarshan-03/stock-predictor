# Stock Predictor

Streamlit app and notebook for Google stock price prediction with an LSTM model.

## Files

- `app.py`: Streamlit UI for uploading CSV data and running predictions.
- `MarketMind_.ipynb`: notebook used for training and exporting the model.
- `data/GOOG.csv`: sample Google stock dataset.
- `requirements.txt`: Python dependencies.

## Run

Create a Python 3.11 virtual environment, install the requirements, then run:

```bash
streamlit run app.py
```

The app expects these generated files:

- `model/stock_model.h5`
- `model/scaler.pkl`
