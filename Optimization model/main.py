##### Library import
# Import Pyomo and the required modules
from pyomo.environ import *

import warnings
import pandas as pd
import numpy as np
from Functions.Data import *
import os
import seaborn as sns
from Functions.Data import *
from Optimisation import *
#pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

def main(Year,
         Location,
         Grid,
         Step,
         Num_interval,
         Ratio,
         SO,
         Batch_interval,
         Hydrogen_storage_type,
         Hydrogen_load_flow,
         Hydrogen_storage_bound,
         Capex_ratio):

    operation_result, key_indicators = optimiser(year=Year,
                                                 location=Location,
                                                 grid=Grid,
                                                 step=Step,
                                                 num_interval=Num_interval, ratio=Ratio,
                                                 SO=SO,
                                                 batch_interval=Batch_interval,
                                                 comp2_conversion=Comp2_conversion(Hydrogen_storage_type),
                                                 hydrogen_storage_type=Hydrogen_storage_type,
                                                 hydrogen_load_flow=Hydrogen_load_flow,
                                                 hydrogen_storage_bound=Hydrogen_storage_bound,
                                                 capex_ratio=Capex_ratio)
    if key_indicators is not None:
        capa = key_indicators['hydrogen_storage_capacity']
        capa = float(capa)
        print(key_indicators)
        return key_indicators,operation_result

    if key_indicators is None:
        print(f'Under the hydrogen storage type {Hydrogen_storage_type}')
        print('No optimal solution found')
        return None, None




'''Parameter input'''
Location='QLD1'           #'QLD1','TAS1','SA1','NSW1','VIC1'
Year=2021
Grid=1
Step=60
Num_interval=0
Ratio=0
SO=1
Batch_interval=24
Hydrogen_storage_type='Lined Rock'              ##'Pipeline','Salt Cavern', 'Lined Rock'
load=180
storage_bound=100
capex_ratio=1
df = pd.DataFrame()
for y in [2021]:
    Year=y
    for L in ['QLD1','TAS1','SA1','NSW1','VIC1']:
        Location = L
        for g in [0,1]:
            Grid=g
            for j in ['Lined Rock','Pipeline']:
                Hydrogen_storage_type=j
                for k in [1,24,720,8760]:
                    Batch_interval=k
                    key_indicators,operation_result=main(Year=Year,Location=Location,Grid=Grid,Step=Step,
                                                     Num_interval=Num_interval,Ratio=Ratio,
                                                     SO=SO,Batch_interval=Batch_interval,
                                                     Hydrogen_storage_type=Hydrogen_storage_type,
                                                     Hydrogen_load_flow=load,
                                                     Hydrogen_storage_bound=storage_bound,
                                                     Capex_ratio=capex_ratio)
                    df = pd.concat([df, key_indicators], ignore_index=True)
                    print(df)
df.to_csv('Result\\different supply periods.csv')

print(df)



















