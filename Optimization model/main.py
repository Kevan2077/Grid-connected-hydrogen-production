##### Library import
# Import Pyomo and the required modules
from pyomo.environ import *

import warnings
import pandas as pd
import numpy as np

from Functions.Data import *
import os

from Functions.Data import *
from Optimisation import *
#pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

def main(Year,
         Location,
         Location_code,
         Grid,
         Opt,
         Step,
         Num_interval,
         Ratio,
         SO,
         Batch_interval,
         storage_type,
         Hydrogen_load_flow,
         Hydrogen_storage_bound,
         bat_class):
    df=pd.DataFrame()
    opt1=pd.DataFrame()
    opt2 = pd.DataFrame()
    if storage_type=='All':
        for i in ['Pipeline','Lined Rock']:
            Hydrogen_storage_type = i
            operation_result, key_indicators = optimiser(year=Year,
                                                 location=Location,
                                                location_code=Location_code,
                                                 grid=Grid,
                                                 opt=Opt,
                                                 step=Step,
                                                 num_interval=Num_interval, ratio=Ratio,
                                                 SO=SO,
                                                 batch_interval=Batch_interval,
                                                 comp2_conversion=Comp2_conversion(Hydrogen_storage_type),
                                                 hydrogen_storage_type=Hydrogen_storage_type,
                                                 hydrogen_load_flow=Hydrogen_load_flow,
                                                 hydrogen_storage_bound=Hydrogen_storage_bound,
                                                 c_bat_class=bat_class)

            if key_indicators is not None:
                print(key_indicators)
                df = pd.concat([df, key_indicators], ignore_index=True)
                if Hydrogen_storage_type=='Pipeline':
                    opt1=pd.concat([opt1, operation_result], axis=1)
                else:
                    opt2 = pd.concat([opt2, operation_result], axis=1)
            if key_indicators is None:
                print(f'Under the hydrogen storage type {Hydrogen_storage_type}')
                print('No optimal solution found')
        if df is None or df.empty:
            print('No optimal solution found under two types of hydrogen storage equipments')
            print(f'temporal correlation {Num_interval} is not feasible, assume the off-grid')
            return None,None

        else: #no solution under hourly temporal correlation
            min_index = df['LCOH'].idxmin()
            # Drop the row with the minimum value in 'LCOH' column
            df.drop(df.index.difference([min_index]), inplace=True)
            if (df['hydrogen_storage_type'] == 'Pipeline').any():
                return df, opt1
            else:
                return df, opt2

    else:
        Hydrogen_storage_type = storage_type
        operation_result, key_indicators = optimiser(year=Year,
                                                     location=Location,
                                                     location_code=Location_code,
                                                     grid=Grid,
                                                     opt=Opt,
                                                     step=Step,
                                                     num_interval=Num_interval, ratio=Ratio,
                                                     SO=SO,
                                                     batch_interval=Batch_interval,
                                                     comp2_conversion=Comp2_conversion(Hydrogen_storage_type),
                                                     hydrogen_storage_type=Hydrogen_storage_type,
                                                     hydrogen_load_flow=Hydrogen_load_flow,
                                                     hydrogen_storage_bound=Hydrogen_storage_bound,
                                                     c_bat_class=bat_class)
        #return all the results given various hydrogen storage options
        return key_indicators,operation_result


'''Parameter input'''
Location='QLD1'           #'QLD1','TAS1','SA1','NSW1','VIC1'
Year=2021
Grid=0
Opt=0      # 0: indicates fixed capacaity; 1: optimized capacity
Step=60
Num_interval=0  # 720,1440,2160,2880,4320,8760,0       1440 means two months interval
Ratio=1
SO=0
Batch_interval=1
Hydrogen_storage_type='Lined Rock'              ##'Pipeline', 'Lined Rock', 'All' (All means choose the one between two options with minimum LCOH)
load=180
storage_bound=120   #tonnes
battery_class='AA'              #["AAA", "AA", "A", "B", "C", "D", "E",'SAM_2020','SAM_2030','SAM_2050']
Location_code='Cell 2126'


'''Read location code information'''

result = pd.DataFrame()
for y in [2023]:
    Year=y
    for i in ['QLD1']:
        if i=='QLD1':
            location_value='Cell 1375'
        elif i == 'SA1':
            location_value='Cell 266'
        elif i =='TAS1':
            location_value='Cell 20'
        elif i =='NSW1':
            location_value='Cell 2127'
        elif i =='VIC1':
            location_value='Cell 36'
        else:
            location_value = i

        Location_code = location_value
        grid_code = i

        print(i)
        print(grid_code)
        Grid=0
        SO=0
        Opt=1
        Hydrogen_storage_type = 'Lined Rock'

        for i in [0]:
            Num_interval=i
            key_indicators,operation_result=main(Year=Year,Location=grid_code,Location_code=Location_code,Grid=Grid,Opt=Opt,Step=Step,
                                                         Num_interval=Num_interval,Ratio=Ratio,
                                                         SO=SO,Batch_interval=Batch_interval,
                                                         storage_type=Hydrogen_storage_type,
                                                         Hydrogen_load_flow=load,
                                                         Hydrogen_storage_bound=storage_bound,
                                                         bat_class=battery_class)
            result = pd.concat([result, key_indicators], ignore_index=True)
            print(result)



















