# Commodities Var Calculations



Commodities VAR calculations for Brent, Gold, Copper, WTI and Henry Hub Natural Gas. Price Data is sources from Yahoo Finance and VAR calculations are done on the rolling front month contract.

This app runs on streamlit thus, you need to create a python virtual environment and install requirements.txt. 

To execute the app, run

```python
python -m streamlit run app.py
```

The app is divided into 4 main sections:

1. VAR using Historical Data
2. VAR using Monte Carlo Simulation
3. VAR using GARCH Model

**NOTE: This is not optimized, nor is production ready. It was created on the fly to demonstrate my knowledge of Value at Risk calculations for commodities. It is not intended to be used in a production environment nor is representative of my programming ethic in general.**

