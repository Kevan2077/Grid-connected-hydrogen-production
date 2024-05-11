##### Library import
# Import Pyomo and the required modules
from pyomo.environ import *
#from sklearn import metrics
import warnings
import pandas as pd
import numpy as np
from Functions.Data import *
import os
import seaborn as sns
from Functions.Data import *
from Optimisation import *
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

'''Parameter input'''
Location='QLD1'           #'QLD1','TAS1','SA1','NSW1','VIC1'
Year=2021
Grid=0
Step=60
Num_interval=0
Ratio=0
SO=0
Batch_interval=1

df = pd.DataFrame()
for y in [2021]:
    Year=y
    for L in ['QLD1','TAS1','SA1','NSW1','VIC1']:
        Location = L
        for i in [24]:
            Batch_interval=i
            key_indicators,operation_result=main(Year=Year,Location=Location,Grid=Grid,Step=Step,Num_interval=Num_interval,Ratio=Ratio,SO=SO,Batch_interval=Batch_interval)
            df = pd.concat([df, key_indicators], ignore_index=True)

df.to_csv('off-grid result.csv')



















