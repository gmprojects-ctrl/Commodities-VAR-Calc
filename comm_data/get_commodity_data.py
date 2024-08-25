# Imports
import pandas as pd
import yfinance as yf


# Commodity list
COMMODITY_LIST = {
    "BZ=F" :  "Brent",  # Brent Crude Oil
    "GC=F" : "Gold",  # Gold
    "HG=F" : "Copper",  # Copper
    "CL=F" : "WTI",  # WTI Crude Oil
    
}

START_DATE = '2010-01-01'


def get_commodity_data():
    comm_data = []
    
    for comm in COMMODITY_LIST:
        data = yf.download(comm, start=START_DATE)
        data['Ticker'] = COMMODITY_LIST.get(comm)
        comm_data.append(data)
    
    comm_data = pd.concat(comm_data,axis=0)
    
    return comm_data

# Main function
if __name__ == "__main__":
    data =get_commodity_data()
    data.to_feather("data.feather")
    