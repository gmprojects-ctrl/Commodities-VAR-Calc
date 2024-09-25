# Imports
import pandas as pd
import yfinance as yf


# Commodity list
COMMODITY_LIST = {
    "BZ=F" :  "Brent",  # Brent Crude Oil
    "GC=F" : "Gold",  # Gold
    "HG=F" : "Copper",  # Copper
    "CL=F" : "WTI",  # WTI Crude Oil
    "NG=F" : "HH Natural Gas", # Henry Hub Natural Gas
    
}

# Start date
START_DATE = '2010-01-01'


def get_commodity_data():
    '''
    Title: get_commodity_data
    Description: This function downloads commodity data from Yahoo Finance and writes it to a feather file in the local directory.
    Parameters: None
    Returns: None
    '''

    comm_data = []
    
    for comm in COMMODITY_LIST:
        data = yf.download(comm, start=START_DATE)
        data['Ticker'] = COMMODITY_LIST.get(comm)
        comm_data.append(data)
    
    comm_data = pd.concat(comm_data,axis=0)

    comm_data.to_feather("data.feather") 

    
    

# Main function
if __name__ == "__main__":
    raise RuntimeError("This script is not meant to be run directly") 