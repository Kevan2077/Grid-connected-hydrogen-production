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
         storage_type,
         Hydrogen_load_flow,
         Hydrogen_storage_bound):
    df=pd.DataFrame()
    opt=pd.DataFrame()
    if storage_type=='All':
        for i in ['Pipeline','Lined Rock']:
            Hydrogen_storage_type = i
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
                                                 hydrogen_storage_bound=Hydrogen_storage_bound)

            if key_indicators is not None:
                print(key_indicators)
                df = pd.concat([df, key_indicators], ignore_index=True)
                opt=pd.concat([opt, operation_result], axis=1)

            if key_indicators is None:
                print(f'Under the hydrogen storage type {Hydrogen_storage_type}')
                print('No optimal solution found')
        min_index = df['LCOH'].idxmin()
        # Drop the row with the minimum value in 'LCOH' column
        df.drop(df.index.difference([min_index]), inplace=True)
        return df, operation_result

    else:
        Hydrogen_storage_type = storage_type
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
                                                     hydrogen_storage_bound=Hydrogen_storage_bound)
        #return all the results given various hydrogen storage options
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
Hydrogen_storage_type='Lined Rock'              ##'Pipeline', 'Lined Rock', 'All' (All means choose the one between two options with minimum LCOH)
load=180
storage_bound=100    #tonnes

df = pd.DataFrame()
for y in [2021]:
    Year=y
    for L in ['QLD1']:
        Location = L
        for j in ['Pipeline']:
            Hydrogen_storage_type=j
            for i in [1,24,720,8760]:
                Batch_interval=i
                key_indicators,operation_result=main(Year=Year,Location=Location,Grid=Grid,Step=Step,
                                                     Num_interval=Num_interval,Ratio=Ratio,
                                                     SO=SO,Batch_interval=Batch_interval,
                                                     storage_type=Hydrogen_storage_type,
                                                     Hydrogen_load_flow=load,
                                                     Hydrogen_storage_bound=storage_bound)
                df = pd.concat([df, key_indicators], ignore_index=True)
                print(df)
                operation_result.to_csv(f'Result\\Flow Track\\QLD different batch interval flow track {i}.csv')

print(df)



















