##### Library import
# Import Pyomo and the required modules
from pyomo.core import Var
from pyomo.environ import *
#from sklearn import metrics
import warnings
import pandas as pd
import numpy as np
from Functions.Data import *
import os
import sys

import seaborn as sns

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

location='QLD1'
year='2021'
step=60
file_name='Optimization model\\Dataset\\'+'Dataframe '+str(location)+'.csv'
file_path = r'{}'.format(os.path.abspath(file_name))
source_df=pd.read_csv(file_path, index_col=0)

'''Electricity Price'''
data=Spotprice(year,location,step)
'''MEF'''
data['Carbon intensity']=carbon_intensity(year,location,step)['Intensity_Index']

'''AEF'''
#data['Mean Carbon intensity']=Mean_carbon_intensity(year,location,step)['Intensity_Index']
data['Mean Carbon intensity'] = source_df['AEF']
'''
When we adopt different time resolution, we can change 'step': 15,30,60 (default) 
'''
new_solar=divide(source_df['Solar'],step)
new_wind=divide(source_df['Wind'],step)
data['Solar']=new_solar
data['Wind']=new_wind
data['Day'] = data['Time'].dt.dayofyear
source_df=data
source_df=source_df[source_df['Day']==365]
source_df.reset_index(drop=True, inplace=True)
print(source_df)