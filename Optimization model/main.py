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
from Optimisation import *
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

'''Parameter input'''
Location='QLD1'
Year=2021
Grid=1
Step=60
Num_interval=0
Ratio=0
SO=0
Batch_interval=1

df=pd.DataFrame()
operation_result,key_indicators=optimiser(year=Year, location=Location,
                                          grid=Grid, step=Step,
                                          num_interval=Num_interval,ratio=Ratio,
                                          SO=SO, batch_interval=Batch_interval)
#path=f'D:\\Do it\\Phd\\Pycharm project\\Grid-connected hydrogen\\Local factory\\Resultset\\{j} Opt.csv'
#operation_result.to_csv(path)
df=pd.concat([df, key_indicators], ignore_index=True)


