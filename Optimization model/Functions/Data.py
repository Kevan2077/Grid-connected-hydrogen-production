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
    file_name = f'{year}.csv'
    file_path = os.path.join(r'D:\Do it\Phd\ECHO\ECHO\NEMED data\Mean carbon intensity', file_name)
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
    file_name = f'{year}.csv'
    file_path = os.path.join(r'D:\Do it\Phd\ECHO\ECHO\NEMED data\Carbon intensity', file_name)
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
    file_name = f'{year}.csv'
    file_path = os.path.join(r'D:\Do it\Phd\ECHO\ECHO\NEMED data\Price', file_name)
    result = pd.read_csv(file_path, index_col=0)
    df = result[result['Region'] == location]

    df['Time'] = pd.to_datetime(df['Time'])
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