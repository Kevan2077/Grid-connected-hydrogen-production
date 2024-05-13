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

def main(Year, Location, Grid, Step, Num_interval, Ratio, SO, Batch_interval, Hydrogen_storage_type,Hydrogen_load_flow):
    df = pd.DataFrame()
    initial_ug_capa = 0
    if Hydrogen_storage_type =='Salt Cavern' or Hydrogen_storage_type =='Lined Rock':
        initial_ug_capa = 110
    if Hydrogen_storage_type =='Pipeline':
        initial_ug_capa=90
    operation_result, key_indicators = optimiser(year=Year,
                                                 location=Location,
                                                 grid=Grid,
                                                 step=Step,
                                                 num_interval=Num_interval, ratio=Ratio,
                                                 SO=SO,
                                                 batch_interval=Batch_interval,
                                                 hydrogen_storage_cost=Cost_hs(initial_ug_capa,
                                                                               Hydrogen_storage_type),
                                                 comp2_conversion=Comp2_conversion(initial_ug_capa),
                                                 hydrogen_storage_type=Hydrogen_storage_type,
                                                 hydrogen_load_flow=Hydrogen_load_flow)
    if key_indicators is not None:
        capa = key_indicators['hydrogen_storage_capacity']
        capa = float(capa)
    if key_indicators is None:
        print(f'Under the hydrogen storage type {Hydrogen_storage_type}')
        print('No optimal solution found')
        return None, None

    if capa > 0 and Hydrogen_storage_type !='Pipeline':        #We assume the capex of unit pipeline storage cost is fixed
        new_ug_capa = capa / 1e3
        if np.mean([new_ug_capa, initial_ug_capa]) > 0:
            while abs(new_ug_capa - initial_ug_capa) / np.mean([new_ug_capa, initial_ug_capa]) > 0.05:
                # Break out of the loop when the condition is met
                initial_ug_capa = new_ug_capa
                print('Refining storage cost; new storage capa=', initial_ug_capa)
                operation_result, key_indicators =  optimiser(year=Year,
                                       location=Location,
                                       grid=Grid,
                                       step=Step,
                                       num_interval=Num_interval,
                                       ratio=Ratio,
                                       SO=SO,
                                       batch_interval=Batch_interval,
                                        hydrogen_storage_cost=Cost_hs(initial_ug_capa,
                                                            Hydrogen_storage_type),
                                       comp2_conversion=Comp2_conversion(
                                           initial_ug_capa),
                                        hydrogen_storage_type=Hydrogen_storage_type,
                                        hydrogen_load_flow=Hydrogen_load_flow)
                capa = key_indicators['hydrogen_storage_capacity']
                capa = float(capa)
                new_ug_capa = capa / 1e3
        df = pd.concat([df, key_indicators], ignore_index=True)
        print(df)
        return df, operation_result
    else:
        print('No iteration')
        print(key_indicators)
        return key_indicators,operation_result


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

df = pd.DataFrame()
for y in [2021]:
    Year=y
    for L in ['QLD1','TAS1','SA1','NSW1','VIC1']:
        Location = L
        for j in ['Pipeline','Lined Rock']:
            Hydrogen_storage_type=j
            key_indicators,operation_result=main(Year=Year,Location=Location,Grid=Grid,Step=Step,Num_interval=Num_interval,Ratio=Ratio,SO=SO,Batch_interval=Batch_interval,Hydrogen_storage_type=Hydrogen_storage_type,Hydrogen_load_flow=load)
            df = pd.concat([df, key_indicators], ignore_index=True)

#df.to_csv('Result\\off-grid different supply periods.csv')
df.to_csv('on-grid test.csv')
print(df)



















