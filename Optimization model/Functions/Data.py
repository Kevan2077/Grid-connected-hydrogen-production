import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, json

def Interp(start, end, step):
    if start == 0 and end == 0:
        size = int(60 / step)
        return np.zeros(size)  # Assuming  elements for the 60-minute interval
    
    time_intervals = np.arange(0, 61, step)  # 0 to 60 minutes in specified intervals

    # Linearly interpolate wind generation using numpy.interp
    interpolated_generation = np.interp(time_intervals, [0, 60], [start, end])

    # Normalize the interpolated values to ensure the sum is equal to end
    normalized_interpolated_generation = (interpolated_generation / np.sum(interpolated_generation[1:]) * end)
    
    return normalized_interpolated_generation[1:]
def divide(df,step):

    #start to divide
    new_data=[]
    if step==60:
        new_data=df
        if not np.any(np.isnan(new_data)):
            print("All values are reassigned") 
        return new_data
    for i in range(len(df) - 1):
        if i==0:
            new_data.insert(0, df[i]/(60/step)) 
        current = df[i]
        next= df[i + 1]
        new_data.extend(Interp(current,next,step).tolist())
    if not np.any(np.isnan(new_data)):
        print("All values are reassigned")    
    return new_data

def Mean_carbon_intensity(year, location,step):
    file_name='Optimization model\\Dataset\\NEMED data\\Mean carbon intensity'
    file_path = r'{}'.format(os.path.abspath(file_name))
    file_name = f'{year}.csv'
    result = pd.read_csv(file_path, index_col=0)

    df = result[result['Region'] == location]

    # Functions cleaning
    df['TimeEnding'] = pd.to_datetime(df['TimeEnding'],format="%d/%m/%Y %H:%M", errors='coerce')
    df.set_index('TimeEnding', inplace=True)

    # Resample and calculate mean
    resample_frequency = '{}T'.format(step)
    Carbon_Intensity = df['Intensity_Index'].resample(resample_frequency).mean()
    Carbon_Intensity = Carbon_Intensity.reset_index()
        

    last_row_index = Carbon_Intensity.index[-1]
    Carbon_Intensity = Carbon_Intensity.drop(last_row_index)
    Carbon_Intensity['Intensity_Index'] = Carbon_Intensity['Intensity_Index'].apply(lambda x: max(x, 0))

    # Fill NaN values in 'Intensity_Index' with the average of previous and next day values at the same 30-minute mark
    Shift=int(60/step*24)
    
    Carbon_Intensity['Intensity_Index'] = Carbon_Intensity['Intensity_Index'].combine_first(
        (Carbon_Intensity['Intensity_Index'].shift(-Shift) + Carbon_Intensity['Intensity_Index'].shift(Shift)) / 2
    )

    # Sort DataFrame back to the original order based on index
    Carbon_Intensity = Carbon_Intensity.sort_values(by='TimeEnding')
    if step == 60:
        Carbon_Intensity = Carbon_Intensity.iloc[:]
    elif step == 30:
        Carbon_Intensity = Carbon_Intensity.iloc[:-1]
    elif step == 15:
        Carbon_Intensity = Carbon_Intensity.iloc[:-3]
    
    return Carbon_Intensity

def carbon_intensity(year, location,step):

    file_name='Optimization model\\Dataset\\NEMED data\\Carbon intensity'
    file_path = r'{}'.format(os.path.abspath(file_name))
    file_name = f'{year}.csv'
    file_path = os.path.join(file_path, file_name)
    result = pd.read_csv(file_path, index_col=0)

    df = result[result['Region'] == location]

    # Functions cleaning
    df['Time'] = pd.to_datetime(df['Time'],format="%d/%m/%Y %H:%M", errors='coerce')

    df.set_index('Time', inplace=True)

    # Resample and calculate mean
    resample_frequency = '{}T'.format(step)
    Carbon_Intensity = df['Intensity_Index'].resample(resample_frequency).mean()
    Carbon_Intensity = Carbon_Intensity.reset_index()
        

    last_row_index = Carbon_Intensity.index[-1]
    Carbon_Intensity = Carbon_Intensity.drop(last_row_index)
    Carbon_Intensity['Intensity_Index'] = Carbon_Intensity['Intensity_Index'].apply(lambda x: max(x, 0))

    # Fill NaN values in 'Intensity_Index' with the average of previous and next day values at the same 30-minute mark
    Shift=int(60/step*24)
    
    Carbon_Intensity['Intensity_Index'] = Carbon_Intensity['Intensity_Index'].combine_first(
        (Carbon_Intensity['Intensity_Index'].shift(-Shift) + Carbon_Intensity['Intensity_Index'].shift(Shift)) / 2
    )

    # Sort DataFrame back to the original order based on index
    Carbon_Intensity = Carbon_Intensity.sort_values(by='Time')
    if step == 60:
        Carbon_Intensity = Carbon_Intensity.iloc[:]
    elif step == 30:
        Carbon_Intensity = Carbon_Intensity.iloc[:-1]
    elif step == 15:
        Carbon_Intensity = Carbon_Intensity.iloc[:-3]
    
    return Carbon_Intensity

def Spotprice(year, location,step):
    file_name='Optimization model\\Dataset\\NEMED data\\Price'
    file_path = r'{}'.format(os.path.abspath(file_name))
    file_name = f'{year}.csv'
    file_path = os.path.join(file_path, file_name)
    result = pd.read_csv(file_path, index_col=0)
    df = result[result['Region'] == location]

    df['Time'] = pd.to_datetime(df['Time'], format='mixed')
    df.set_index('Time', inplace=True)

    # Resample and calculate 15-minute mean
    resample_frequency = '{}T'.format(step)
    prices = df['Prices'].resample(resample_frequency).mean()
    prices = prices.reset_index()

    last_row_index = prices.index[-1]
    prices = prices.drop(last_row_index)
    prices['Prices'] = prices['Prices'].apply(lambda x: max(x, 0))
    prices['Prices'] = prices['Prices'] / 1000

    # Fill NaN values in 'Prices' with the average of previous and next day values at the same 15-minute mark
    Shift=int(60/step*24)
    
    prices['Prices'] = prices['Prices'].combine_first(
        (prices['Prices'].shift(-Shift) + prices['Prices'].shift(Shift)) / 2
    )

    # Sort DataFrame back to the original order based on index
    prices = prices.sort_values(by='Time')
    
    if step == 60:
        prices = prices.iloc[:]
    elif step == 30:
        prices = prices.iloc[:-1]
    elif step == 15:
        prices = prices.iloc[:-3]
    
    return prices

'''
def Cost_hs(size,storage_type):
    if size > 0:
        x = np.log10(size)
        #if size >= 100:
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
    #else:
        if storage_type == 'Pipeline':
            print('storage_type is Pipeline Storage')
            # cost = 10 ** (-0.0285*x + 2.7853)
            cost = 516
    else:
        print('No storage set up')
        cost = 516
    return (cost)
'''


def Comp2_conversion(hydrogen_storage_type):
    if hydrogen_storage_type=='Lined Rock' or  hydrogen_storage_type=='Salt Cavern':
        comp2_conversion = 0.41
    else: #all the hydrogen pumped into pipeline storage, in this case, no need to pump again.
        comp2_conversion = 0
    return (comp2_conversion)


def piecewise_function(upper_bound, insert_point,hydrogen_storage_type):
    cross_point=21.74214531
    if hydrogen_storage_type=='Pipeline':
        pipeline_range = np.linspace(0, cross_point, insert_point)
        x = np.log10(pipeline_range)
        cost = 10 ** (-0.0285 * x + 2.7853)
        cost[0] = 1000
        return pipeline_range * 1000, cost

    if hydrogen_storage_type=='Lined Rock':
        LRC_range = np.linspace(cross_point,upper_bound, insert_point)
        cost = np.where(
            LRC_range <= 4000,
            10 ** (0.217956 * np.log10(LRC_range) ** 2 - 1.575209 * np.log10(LRC_range) + 4.463930),
            41.48
        )

        return LRC_range, cost


def c_bat_cost(cost_class):
    c_bat= {
        "Cost Class": ["AAA", "AA", "A", "B", "C", "D", "E",'SAM_2020','SAM_2030','SAM_2050'],
        "c_bat_e": [20, 40, 58, 73, 86, 101, 116,197,164,131],
        "c_bat_p": [217, 441, 658, 822, 987, 1152, 1316,405,338,270]
    }

    c_bat = pd.DataFrame(c_bat)

    # Get the row corresponding to the cost class
    result = c_bat[c_bat["Cost Class"] == cost_class]
    c_bat_e=result['c_bat_e'].values[0]
    c_bat_p=result['c_bat_p'].values[0]
    return c_bat_e,c_bat_p