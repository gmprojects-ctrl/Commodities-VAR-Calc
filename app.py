# Imports
import pandas as pd
import streamlit as st 
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go   
from scipy.stats import norm

from comm_data import COMMODITY_LIST

# Data Path
DATA_PATH = Path("./data.feather")


# Get the log returns
def get_log_returns(dataframe:pd.DataFrame):
    # Shift the return by 1
    previous_returns = dataframe.shift(1)   
    # Convert into numpy arrays
    previous_returns = previous_returns.to_numpy()
    dataframe_ = dataframe.to_numpy()
    
    # Get the returns and flatten it
    log_returns = np.log((dataframe_ / previous_returns))
    
    # Drop the last value
    log_returns = log_returns[1:]
    
    
    # Create a pandas dataframe
    log_returns = pd.DataFrame(log_returns,index=dataframe.index[:-1],columns=['Log Returns'])
    
    return log_returns

# Get the value at risk and expected shortfall
def get_var_cvar(dataframe:pd.DataFrame,rolling_window:int=7,p_value:float=0.05):
    
    # Calculate the log returns
    log_returns = get_log_returns(dataframe)
    
    # Sum the log returns
    sum_log_returns = log_returns.rolling(rolling_window).sum().dropna()
    
    # Calculate the value at risk
    log_value_at_risk = np.quantile(sum_log_returns['Log Returns'],p_value)
    
    # Calculate the expected shortfall
    log_expected_shortfall = sum_log_returns[sum_log_returns['Log Returns'] <= log_value_at_risk].mean().values[0]
    
    return log_value_at_risk,log_expected_shortfall
# Do a Black Sholes Model log_value_at_risk,log_expected_shortfall

def call_black_scholes(S_0:float, K:float, log_vol:float,rate:float, delta_t: float)->float:
    
    # Denominator
    scale = 1/(log_vol * np.sqrt(delta_t))
    
    # dup
    d_up = scale * ( np.log(S_0/K) + delta_t*( rate + 0.5*(log_vol**2)   ))
    
    # d_down
    d_down = d_up - log_vol * np.sqrt(delta_t)
    
    # Discount rate
    discount_rate = np.exp(-rate*(delta_t))
    
    # D_up
    
    d_up_cum = norm().cdf(d_up)
    
    d_down_cum = norm().cdf(d_down)
    
    # Call value
    call_value = S_0 * d_up_cum - K*discount_rate*d_down_cum
    
    # Return
    
    return call_value
    
def put_black_scholes(S_0:float, K:float, log_vol:float,rate:float, delta_t: float)->float:
    # Get the call value
    call_value = call_black_scholes(S_0, K, log_vol,rate, delta_t)
    
    # Put value
    return 0
    
    
    

# Main function
def main():
    st.title("Portfolio Analyzer")
    
    # Load the data
    try:
       data = pd.read_feather(DATA_PATH)
       data.index = pd.to_datetime(data.index)
    except Exception as e:
        st.error(f"Cannot load data : {str(e)}")
        raise e 
    
    # Create a data editor
    st.markdown("### Select a Commodity")
    
    commodity = st.selectbox(label="Select a Commodity",options=COMMODITY_LIST.values())
    
    # Enter a period to calculate the value at risk
    period = st.number_input(label="Enter a look back period",min_value=1,step=1,value=365)
    
    # Filter the data
    commodity_data = data[data['Ticker'] == commodity].copy()
    
    # Filter by time
    commodity_data = commodity_data.iloc[-period:,:].copy()
    
    # Sort the data
    commodity_data = commodity_data.sort_index(ascending=True).copy()
    
     
    with st.expander("See data"):
        # Write that data
        st.write(commodity_data)
        
        # Plot the data
        st.plotly_chart(px.line(commodity_data,x=commodity_data.index,y='Close',title=f"{commodity} Prices"))
        

    with st.expander("VAR and Options Price"):
        
        
        
        # Calculate the log returns
        log_returns = get_log_returns(commodity_data['Close'])
        
        # Get a rolling window
        rolling_window = st.number_input(label="Enter a rolling window",min_value=1,step=1,value=7)
        
        # Sum the log returns
        sum_log_returns = log_returns.rolling(rolling_window).sum().dropna()
        
        
        
        # Plot the data
        st.plotly_chart(px.line(sum_log_returns,x=sum_log_returns.index,y='Log Returns',title=f"{commodity} Log Returns over {rolling_window} day rolling sum"))
        
        # Plot the histogram
        st.plotly_chart(px.histogram(sum_log_returns,x='Log Returns',title=f"{commodity} Log Returns Histogram over {rolling_window} day rolling sum", nbins=100))
        
        # Calculate the value at risk
        log_value_at_risk = np.quantile(sum_log_returns['Log Returns'],0.05)
        value_at_risk = np.exp(log_value_at_risk)
        
        # Calculate the expected shortfall
        log_expected_shortfall = sum_log_returns[sum_log_returns['Log Returns'] <= log_value_at_risk].mean().values[0]
        expected_shortfall = np.exp(log_expected_shortfall)
        
        # Write the value at risk
        st.write(f"The value at risk is {(1-value_at_risk)*100:.2f}% at the 5% confidence level over a {rolling_window} day period") 
        st.write(f"The expected shortfall is {(1-expected_shortfall)*100:.2f}% at the 5% confidence level over a {rolling_window} day period")

        
        
        # Calculate volatility
        volatility = log_returns.std().values[0] * np.sqrt(252)
        
        # Write the annualized volatility of log returns
        st.write(f"The annualized volatility of log returns is {volatility:.2f}")
        
        # Enter a strike price
        strike_price = st.number_input(label="Enter a strike price",min_value=0.0,step=0.01,value=1000.0)
        
        # Enter a risk free rate
        rate = st.number_input(label="Enter a risk free rate",min_value=0.0,step=0.01,value=0.01) / 100
        
        # Enter a time to expiration
        time_to_expiration = st.number_input(label="Enter a time to expiration",min_value=0,step=1,value=1)

        # Divide by 12
        delta_t = time_to_expiration/12
        
        # Calculate the call price
        call_price = call_black_scholes(commodity_data['Close'].iloc[-1],strike_price,volatility,rate,delta_t)
        
        # Write the call price
        st.write(f"The call price is {call_price:.2f}")
        


# Run the function
if __name__ == "__main__":
    main()