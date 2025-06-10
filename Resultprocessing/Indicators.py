import numpy as np
import pandas as pd
import calendar
import warnings



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
    df['electrolyser_capex+electrolyser_OM']=(df['electrolyser_capacity']*(1343.3)*CRF+df['electrolyser_capacity']*(37.4))/df['production_amount']+0.02
    df['wind_OM']=df['wind_capacity']*17.5/df['production_amount']
    df['pv_OM']=df['pv_capacity']*(11.9)/df['production_amount']
    df['electrolyser_OM']=df['electrolyser_capacity']*(37.4)/df['production_amount']+0.02
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

'''Batch periods'''


def EI_batch_periods(location, operation_result, interval):
    production_results = []
    carbon_results = []
    sell_results = []
    purchase_results = []
    grid_cost_results = []
    # Iterate over the dataframe in chunks of {interval} rows
    if interval != 720:
        for i in range(0, len(operation_result), interval):
            # Get the chunk of rows
            chunk = operation_result.iloc[i:i + interval]
            # Check if the length of the chunk matches the specified interval length
            # Calculate the sum for each column in the chunk
            production_sum = -chunk['el_pout'].sum()
            carbon_sum = chunk['MEF_CO2'].sum()
            sell_sum = chunk['grid_pin'].sum()
            purchase_sum = chunk['grid_pout'].sum()
            grid_cost_sum = chunk['grid_electricity_cost'].sum()
            # Append the sums to the respective results lists
            production_results.append(production_sum)
            carbon_results.append(carbon_sum)
            sell_results.append(sell_sum)
            purchase_results.append(purchase_sum)
            grid_cost_results.append(grid_cost_sum)

    else:
        print('Interval is monthly resolution')
        supply_points = [0, 24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30, 24 * 31,
                         24 * 30, 24 * 31]
        cumulative_supply_points = np.cumsum(supply_points)

        for i in range(0, len(cumulative_supply_points) - 1):
            chunk = operation_result.iloc[cumulative_supply_points[i]:cumulative_supply_points[i + 1]]
            production_sum = -chunk['el_pout'].sum()
            carbon_sum = chunk['MEF_CO2'].sum()
            sell_sum = chunk['grid_pin'].sum()
            purchase_sum = chunk['grid_pout'].sum()
            grid_cost_sum = chunk['grid_electricity_cost'].sum()
            # Append the sums to the respective results lists
            production_results.append(production_sum)
            carbon_results.append(carbon_sum)
            sell_results.append(sell_sum)
            purchase_results.append(purchase_sum)
            grid_cost_results.append(grid_cost_sum)

    batch_df = pd.DataFrame({'Batch': range(1, len(production_results) + 1),
                             'Total_production': production_results,
                             'Total_CO2': carbon_results,
                             'Total_sell': sell_results,
                             'Total_purchase': purchase_results,
                             'Total_grid_cost': grid_cost_results})
    # batch_df['Marginal_cost'] = batch_df['Total_grid_cost']/1.49 / batch_df['Total_production']+0.075
    batch_df['EI_MEF'] = np.where(batch_df['Total_production'] == 0, 0,
                                  batch_df['Total_CO2'] / batch_df['Total_production'])
    batch_df['EI_Market'] = np.where(batch_df['Total_production'] == 0, 0,
                                     (-1 * batch_df['Total_purchase'] * (1 - 0.188) - (batch_df['Total_sell'])) * 0.81 /
                                     batch_df['Total_production'])

    if location == 'QLD1':
        batch_df['EI_L'] = np.where(batch_df['Total_production'] == 0, 0,
                                    -1 * (batch_df['Total_purchase'] + batch_df['Total_sell']) * 0.73 / batch_df[
                                        'Total_production'])
    if location == 'NSW1':
        batch_df['EI_L'] = np.where(batch_df['Total_production'] == 0, 0,
                                    -1 * (batch_df['Total_purchase'] + batch_df['Total_sell']) * 0.68 / batch_df[
                                        'Total_production'])
    if location == 'TAS1':
        batch_df['EI_L'] = np.where(batch_df['Total_production'] == 0, 0,
                                    -1 * (batch_df['Total_purchase'] + batch_df['Total_sell']) * 0.12 / batch_df[
                                        'Total_production'])
    if location == 'VIC1':
        batch_df['EI_L'] = np.where(batch_df['Total_production'] == 0, 0,
                                    -1 * (batch_df['Total_purchase'] + batch_df['Total_sell']) * 0.79 / batch_df[
                                        'Total_production'])
    if location == 'SA1':
        batch_df['EI_L'] = np.where(batch_df['Total_production'] == 0, 0,
                                    -1 * (batch_df['Total_purchase'] + batch_df['Total_sell']) * 0.25 / batch_df[
                                        'Total_production'])
    batch_df.replace([np.inf, -np.inf], 0, inplace=True)
    # When RECs is positive, which indicate we obtain RECs by selling more electricity
    batch_df['RECs'] = (batch_df['Total_purchase'] * (1 - 0.188) + batch_df['Total_sell']) / 1000
    return batch_df








