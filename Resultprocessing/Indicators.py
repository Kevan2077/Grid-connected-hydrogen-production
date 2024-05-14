import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import calendar
import warnings
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.dates import MonthLocator


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def Cost_hs(size,storage_type):
    if size > 0:
        x = np.log10(size)
        if size >= 100:
            if storage_type=='Salt Cavern':
                print('Storage_type is Salt Cavern')
                cost = 10 ** (0.212669 * x ** 2 - 1.638654 * x + 4.403100)
                if size > 8000:
                    cost = 17.66
            if storage_type == 'Lined Rock':
                print('Storage_type is Lined Rock')
                cost = 10 ** (0.217956 * x ** 2 - 1.575209 * x + 4.463930)
                if size > 4000:
                    cost = 41.48
        else:
            print('storage_type is Pipeline Storage')
            # cost = 10 ** (-0.0285*x + 2.7853)
            cost = 516
    else:
        print('No storage set up')
        cost = 516
    return (cost)


#Add the cost compositionin formation
def Calculation_LCOH(df):
    CRF=0.07822671821
    df['wind_capex+wind_OM']=(df['wind_capacity']*2126.6*CRF+df['wind_capacity']*17.5)/df['production_amount']
    df['pv_capex+pv_OM']=(df['pv_capacity']*(1068.2)*CRF+df['pv_capacity']*(11.9))/df['production_amount']
    df['hydrogen_storage_capex']=df['hydrogen_storage_capacity']*df['hydrogen_storage_cost']*CRF/df['production_amount']
    df['electrolyser_capex+electrolyser_OM']=(df['electrolyser_capacity']*(1343)*CRF+df['electrolyser_capacity']*(37.4))/df['production_amount']+0.075
    df['wind_OM']=df['wind_capacity']*17.5/df['production_amount']
    df['pv_OM']=df['pv_capacity']*(11.9)/df['production_amount']
    df['electrolyser_OM']=df['electrolyser_capacity']*(37.4)/df['production_amount']+0.075
    df['grid_electricity_cost']=df['grid_cost']*0.7/(df['production_amount'])
    df['LCOH_sum'] = (df['wind_capex+wind_OM'] +
                   df['pv_capex+pv_OM'] +
                   df['hydrogen_storage_capex'] +
                   df['electrolyser_capex+electrolyser_OM'] +df['grid_electricity_cost'])
    return df
'''
def Calculation_LCOH(df):
    CRF=0.07822671821
    df['wind_capex']=df['wind_capacity']*1455*CRF/df['production_amount']
    df['pv_capex']=df['pv_capacity']*(1122.7)*CRF/df['production_amount']
    df['hydrogen_storage_capex']=df['hydrogen_storage_capacity']*Cost_hs(df['hydrogen_storage_capacity'],df['hydrogen_storage_type'])*CRF/df['production_amount']
    df['electrolyser_capex']=df['electrolyser_capacity']*(1067)*CRF/df['production_amount']
    df['wind_OM']=df['wind_capacity']*18.65/df['production_amount']
    df['pv_OM']=df['pv_capacity']*(12.7)/df['production_amount']
    df['electrolyser_OM']=df['electrolyser_capacity']*(37.4)/df['production_amount']+0.075
    df['grid_electricity_cost']=df['grid_cost']*0.7/(df['production_amount'])
    df['LCOH_sum'] = (df['wind_capex'] + df['wind_OM'] +
                   df['pv_capex'] + df['pv_OM'] +
                   df['hydrogen_storage_capex'] +
                   df['electrolyser_capex'] + df['electrolyser_OM'] +df['grid_electricity_cost'])
    return df
'''


