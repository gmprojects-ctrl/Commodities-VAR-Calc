# Imports
import pandas as pd
import streamlit as st 
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go   
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf
import arch

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
    
     # Calculate the log returns
    log_returns = get_log_returns(commodity_data['Close'])
        
    # Get a rolling window
    rolling_window = st.number_input(label="Enter a rolling window",min_value=1,step=1,value=7)
        
    # Sum the log returns
    sum_log_returns = log_returns.rolling(rolling_window).sum().dropna()
    
    
    with st.expander("See data"):
        # Write that data
        st.write(commodity_data)
        
        # Plot the data
        st.plotly_chart(px.line(commodity_data,x=commodity_data.index,y='Close',title=f"{commodity} Prices"))
        

    with st.expander("VAR and Options Price"):
        
        
        

        
        
      
        
        # Calculate the value at risk
        log_value_at_risk = np.quantile(sum_log_returns['Log Returns'],0.05)
        value_at_risk = np.exp(log_value_at_risk)
        
        # Calculate the expected shortfall
        log_expected_shortfall = sum_log_returns[sum_log_returns['Log Returns'] <= log_value_at_risk].mean().values[0]
        expected_shortfall = np.exp(log_expected_shortfall)
        
        
          
        # Plot the data
        st.plotly_chart(px.line(sum_log_returns,x=sum_log_returns.index,y='Log Returns',title=f"{commodity} Log Returns over {rolling_window} day rolling sum"))
        
        # Plot the histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sum_log_returns['Log Returns'],nbinsx=50))
        fig.add_vline(x=log_value_at_risk,line=dict(color='red'),name="Value at Risk")  
        fig.add_vline(x=log_expected_shortfall,line=dict(color='green'),name="Expected Shortfall")
        
        fig.update_layout(title=f"Histogram of Log Returns for {commodity} over {rolling_window} day period")
        
        st.plotly_chart(fig)
        
        
        # Write the value at risk
        st.write(f"The value at risk is {(1-value_at_risk)*100:.2f}% at the 5% confidence level over a {rolling_window} day period") 
        st.write(f"The expected shortfall is {(1-expected_shortfall)*100:.2f}% at the 5% confidence level over a {rolling_window} day period")

        
    
    
    with st.expander("Options Pricing"):
        
        # Calculate the daily volatility
        daily_vol = log_returns.std().values[0]
        
        
        # Calculate volatility
        annual_vol = daily_vol * np.sqrt(252)
        
        # Write the daily volatility of log returns
        st.write(f"The daily volatility of log returns is {daily_vol:.2f}")
        
        # Write the annualized volatility of log returns
        st.write(f"The annualized volatility of log returns is {annual_vol:.2f}")
        
        # Enter a strike price
        strike_price = st.number_input(label="Enter a strike price",min_value=0.0,step=0.01,value=1000.0)
        
        # Enter a risk free rate
        rate = st.number_input(label="Enter a risk free rate",min_value=0.0,step=0.01,value=0.01) / 100
        
        # Enter a time to expiration
        time_to_expiration = st.number_input(label="Enter a time to expiration",min_value=0,step=1,value=1)

        # Divide by 12
        delta_t = time_to_expiration/12
        
        # Calculate the call price
        call_price = call_black_scholes(commodity_data['Close'].iloc[-1],strike_price,annual_vol,rate,delta_t)
        
        # Write the call price
        st.write(f"The call price is {call_price:.2f}")
        
    with st.expander(f"Using GARCH model for {commodity} log returns over {rolling_window} day period"):
        st.markdown("""
        ## GARCH Model
        
        Consider the following model:
        $$y_t = \mu + \epsilon_t$$
        
        Where $y_t$ is the log returns, $\mu$ is the mean and $\epsilon_t = \sigma_t * z_t$ where $z_t$ is $N(0,1)£.
        
        Where $y_t$ is the log returns, $\mu$ is the mean, $\sigma_t$ is the volatility and $z_t$ is N(0,1).
        
        Then the GARCH(p,q) model is given by:
        """)
        
        st.markdown(r'$$\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \epsilon_{t-i}^2 + \sum_{i=1}^{q} \beta_i \sigma_{t-i}^2$$')
        
        
        # Garch data
        # Scale by 1000
        garch_data = 1000*(sum_log_returns)
        
        
        # Create a 80/20 split
        training_data = garch_data.iloc[:int(len(garch_data)*0.8)].copy()
        testing_data = garch_data.iloc[int(len(garch_data)*0.8):].copy()
        conditional_training_data = training_data.copy()    
        
        # Divider
        st.divider()
        
        # Note the log returns are mulitplied by a 1000
        st.markdown("Note the log returns are mulitplied by a 1000")
        
        # Create tabs
        tabs = st.tabs(["Training Data","Testing Data"])
        
        # Show the training data
        with tabs[0]:
            st.write(training_data)
        
        # Show the testing data
        with tabs[1]:
            st.write(testing_data)
            
        # PACF plot 
        pacf_plot = pacf(training_data['Log Returns']**2,nlags=20)
        
        # ACF plot
        acf_plot = acf(training_data['Log Returns']**2,nlags=20)
        
        # Create a figure
        fig = go.Figure()
        # Add the PACF plot
        fig.add_trace(go.Bar(x=np.arange(len(pacf_plot)),y=pacf_plot,name="PACF"))
        
        # Add the ACF plot
        fig.add_trace(go.Bar(x=np.arange(len(acf_plot)),y=acf_plot,name="ACF"))
        
        # Set the xaxis
        fig.update_xaxes(title="Lag Period")
        
        # Set the yaxis
        fig.update_yaxes(title="Correlation")
        
        # Update the layout
        fig.update_layout(title="PACF and ACF plot for GARCH model (note the data is squared as it is sigma^2)")
        
        
        # Plot the figure
        st.plotly_chart(fig)
        
        
        # P,Q input
        p_garch = st.number_input(label="Enter a p value for the GARCH model",min_value=0,step=1,value=1)
        q_garch = st.number_input(label="Enter a q value for the GARCH model",min_value=0,step=1,value=1)

        # Create an Arch Model
        model = arch.arch_model(training_data['Log Returns'],mean='Zero',vol='Garch',p=p_garch,q=q_garch)
        
        # Fit the model
        model_fit = model.fit()
        
        # Get the conditional volatility (which is the square root of the variance and equal to the |log returns|)
        conditional_volatility = model_fit.conditional_volatility
        
        # Plot the conditional volatility
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=training_data.index,y=conditional_volatility,name="Conditional Volatility", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=training_data.index,y=np.abs(training_data['Log Returns']),name="Log Returns"))
        fig.update_layout(title=f"Model Fit for GARCH({p_garch,q_garch}) for {commodity} over {rolling_window} day period on Training Data")
        st.plotly_chart(fig)
        
        # Get the forecast
        forcasted_values = pd.DataFrame(index=testing_data.index,columns=['Forecasted Volatility'])
        
        for i in range(len(testing_data)):
            # Get prediction
            prediction = model_fit.forecast(horizon=1).variance.iloc[0].values[0]
            
            # Forecast
            forcasted_values.iloc[i] = np.sqrt(prediction)
            
            # Update the model
            conditional_training_data = pd.concat([conditional_training_data,testing_data.iloc[i:i+1,:]])
            
            # Update the model
            model = arch.arch_model(conditional_training_data['Log Returns'],mean='Zero',vol='Garch',p=p_garch,q=q_garch)
            
            # Fit the model
            model_fit = model.fit()
            
        
            
        # Plot the forecasted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=testing_data.index,y=forcasted_values['Forecasted Volatility'],name="Forecasted Volatility",line=dict(color='red')))
        fig.add_trace(go.Scatter(x=testing_data.index,y=np.abs(testing_data['Log Returns']),name="Log Returns (absolute value)"))
        fig.update_layout(title=f"Rolling Forecasted Volatility ({p_garch,q_garch}) for {commodity} over {rolling_window} day period on Testing Data")
        st.plotly_chart(fig)
        
        # Root mean squared error
        rmse = np.sqrt(np.mean((forcasted_values['Forecasted Volatility'] - np.abs(testing_data['Log Returns']))**2))
        
        
        
        # Write the RMSE
        st.write(f"The RMSE is {rmse:.2f}")
        
        # Write the model summary
        st.write(model_fit.summary())
        
        # Write the model parameters
        st.write(model_fit.params)
        
        

# Run the function
if __name__ == "__main__":
    main()