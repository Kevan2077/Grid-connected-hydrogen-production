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


''' Initialize the optimisation model '''
def optimiser(year, location,location_code, grid, opt, step, num_interval,ratio,SO, batch_interval,
              comp2_conversion,hydrogen_storage_type,hydrogen_load_flow, hydrogen_storage_bound,c_bat_class):

    #data import
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
    '''
    new_solar=divide(source_df['Solar'],step)
    new_wind=divide(source_df['Wind'],step)
    data['Solar']=new_solar
    data['Wind']=new_wind
    '''
    '''Obtain renewable energy generation'''
    pv_path = f'Optimization model\\Dataset\\Renewable generation\\{location_code}_{year}_PV.csv'
    wind_path = f'Optimization model\\Dataset\\Renewable generation\\{location_code}_{year}_wind.csv'

    pv_ref = pd.read_csv(pv_path, index_col=0)
    wind_ref = pd.read_csv(wind_path, index_col=0)

    data['Solar'] = pv_ref['Solar']
    data['Wind'] = wind_ref['Wind']


    source_df=data
    end_index=60/step*8759

    '''Pyomo Model'''
    m = ConcreteModel()
    '''Set the operation time period'''
    num_simulation = len(source_df)
    m.time_periods = RangeSet(0,num_simulation-1)  #default 8760
    print(f'Time resolution: {num_simulation}')

    '''Set the temporal correlation check interval'''
    interval = num_interval
    m.check_periods=RangeSet(0,num_simulation-1,interval)
    m.interval=RangeSet(0,num_simulation)

    '''Set different supply requirement'''
    #larger batch size needs to meet larger hydrogen demand
    m.supply_periods=RangeSet(0,num_simulation-1,batch_interval)
    number_of_supply: int = len(m.supply_periods)
    print('Batch interval:', batch_interval)

    # Set the time period of each calendar month
    supply_periods = [0, 24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30, 24 * 31,
                      24 * 30, 24 * 31]
    cumulative_supply = np.cumsum(supply_periods) #obtain the time point of last hour in each calendar month

    '''initial value of cost, unit: USD'''

    m.c_hydrogen_storage = Var() #piecewise function to determine cost based on its size

    m.c_pv = Param(initialize=1068.2)  # CAPEX of pv
    m.c_wind = Param(initialize=2126.6)  # CAPEX of wind
    m.c_el = Param(initialize=1343.3)  # CAPEX of electrolyser
    c_bat_e, c_bat_p = c_bat_cost(c_bat_class)

    m.c_bat_e = Param(initialize=c_bat_e)  # Unit cost of battery storage USD/kWh
    m.c_bat_p = Param(initialize=c_bat_p)  # Unit cost of battery power capacity USD/kW

    m.CRF = Param(initialize=0.07822671821)
    m.pv_FOM = Param(initialize=11.9)
    m.wind_FOM = Param(initialize=17.5)
    m.el_FOM = Param(initialize=37.4)
    m.el_VOM = Param(initialize=0.02)  # the cost of water consumption

    ''' renewable generation '''
    m.pv_ref_size = Param(initialize=1000)
    m.wind_ref_size = Param(initialize=320000)
    m.pv_ref_generation = Param(m.time_periods,
                                initialize={i: float(source_df.loc[i, 'Solar']) for i in m.time_periods})
    m.wind_ref_generation = Param(m.time_periods,
                                initialize={i: float(source_df.loc[i, 'Wind']) for i in m.time_periods})

    ''' Electricity consumption of compression'''
    m.comp2_power_conversion=Param(initialize=comp2_conversion)

    ''' price and carbon intensity '''
    print(source_df['Prices'])
    m.price = Param(m.time_periods, initialize={i: float(source_df.loc[i, 'Prices']) for i in m.time_periods})
    m.MEF = Param(m.time_periods,
                               initialize={i: float(source_df.loc[i, 'Carbon intensity']) for i in m.time_periods})

    m.AEF = Param(m.time_periods,
                               initialize={i: float(source_df.loc[i, 'Mean Carbon intensity']) for i in
                                                m.time_periods})


    '''
    path=f'Optimization model\\Dataset\\NEMED data\\Mean carbon intensity\\{year} AEF\\{location[:-1]} new.csv'
    AEF=pd.read_csv(path, index_col=0)
    m.AEF = Param(m.time_periods,
                               initialize={i: float(AEF.loc[i, 'New_Intensity_Index']) for i in
                                                m.time_periods})
    print(AEF['New_Intensity_Index'])
    '''
    '''Variable definition'''

    #Indicators:
    m.grid_electricity_cost=Var()
    m.capex=Var()
    m.LCOH=Var()
    m.production_amount=Var()


    #load and total production amount
    m.Load=Var(m.time_periods, domain=NonNegativeReals)

    #check whether the flow can be active or not using Big M method
    m.is_grid_pin_active = Var(m.time_periods, within=Binary)
    m.is_grid_pout_active = Var(m.time_periods, within=Binary)
    m.M=Param(initialize=1e7)

    #Transmission cost may varied according to the maximum power integration scale
    if grid==1:
        print("On-Grid")
        m.maximum_power_integration=Var(domain=NonNegativeReals) #let the system decide

    else:
        print("Off-Grid")
        m.maximum_power_integration=Param(initialize=0)

    #Variable capacity
    #m.pv_capacity=Var(domain=NonNegativeReals)
    #m.wind_capacity=Var(domain=NonNegativeReals)
    #m.electrolyser_capacity=Var(domain=NonNegativeReals)


    bat=0
    if bat==0:
        print('No battery is taken into account')
        m.bat_e_capacity = Param(initialize=0)
        m.bat_p_capacity = Param(initialize=0)
    else:
        print("Battery storage class is ", c_bat_class)
        m.bat_e_capacity = Var(domain=NonNegativeReals)
        m.bat_p_capacity = Var(domain=NonNegativeReals)

    if hydrogen_storage_type=='Pipeline':
        cross_point = 21.74214531
        m.h2_storage_capacity=Var(domain=NonNegativeReals)
        m.h2_storage_capacity_t=Var(domain=NonNegativeReals,bounds=(0, cross_point))
    else:
        cross_point = 21.74214531
        m.h2_storage_capacity = Var(domain=NonNegativeReals)
        m.h2_storage_capacity_t=Var(domain=NonNegativeReals,bounds=(cross_point, hydrogen_storage_bound))

    #Fixed capacity
    #input the off-grid optimized results:
    #file_name='Optimization model\\Result\\Hourly supply periods\\'+'off-grid result'+'.csv'
    '''
    file_name=f'Result\Hourly supply period\grid node calculation\{location_code}.csv'
    file_path = r'{}'.format(os.path.abspath(file_name))
    off_grid_result = pd.read_csv(file_path, index_col=0)

    Opt_off_grid = off_grid_result[off_grid_result['Location'] == location].reset_index(drop=True)
    print(Opt_off_grid)
    '''
    #file_name = f'Result\Hourly supply period\grid node calculation\{location_code}.csv'
    file_name =f'Result\Hourly supply period\Renewableninja\off_grid results_{year}.csv'
    file_path = r'{}'.format(os.path.abspath(file_name))
    off_grid_result = pd.read_csv(file_path, index_col=0)
    Opt_off_grid = off_grid_result[off_grid_result['Location'] == location].reset_index(drop=True)

    #Opt_off_grid = off_grid_result


    if grid == 1:
        print("Location code:",location_code)
        print("Grid:",location)
        print('Capex_limit is open')
        m.capex_limit = Constraint(expr=m.capex <= Opt_off_grid.loc[0, 'Capex'])
        if opt == 0:
            print('No capacity optimization')
            m.pv_capacity = Param(initialize=Opt_off_grid.loc[0, 'pv_capacity'] * ratio)
            m.wind_capacity = Param(initialize=Opt_off_grid.loc[0, 'wind_capacity'] * ratio)
            #m.h2_storage_capacity = Param(initialize=Opt_off_grid.loc[0, 'hydrogen_storage_capacity'])
            m.electrolyser_capacity = Param(initialize=Opt_off_grid.loc[0, 'electrolyser_capacity'])  # 175kw
        else:
            presolved_solution = {'var1': Opt_off_grid.loc[0, 'pv_capacity'],
                                  'var2': Opt_off_grid.loc[0, 'wind_capacity'],
                                  'var3': Opt_off_grid.loc[0, 'electrolyser_capacity'],
                'var4':Opt_off_grid.loc[0, 'hydrogen_storage_capacity']}
        # Initialize Pyomo variables with presolved values
            m.pv_capacity = Var(initialize=presolved_solution['var1'],domain=NonNegativeReals)
            m.wind_capacity = Var(initialize=presolved_solution['var2'],domain=NonNegativeReals)
            m.electrolyser_capacity = Var(initialize=presolved_solution['var3'],domain=NonNegativeReals)
            #m.h2_storage_capacity=Var(initialize=presolved_solution['var4'],domain=NonNegativeReals)



    print(f'State: {location}')
    print(f'Location_code: {location_code}')
    '''Flow variables'''

    #PV and wind node:
    m.pv_pout = Var(m.time_periods, domain=NonPositiveReals)  #power out of PV plant (kW)
    m.wind_pout = Var(m.time_periods, domain=NonPositiveReals)  #power out of wind farm (kW)
    m.CP_curtailment = Var(m.time_periods, within=NonPositiveReals)  #curtailed power (kW)

    # Battery node
    m.bat_pin = Var(m.time_periods, domain=NonNegativeReals)  # power in of battery plant (kW)
    m.bat_pout=Var(m.time_periods, domain=NonPositiveReals)  # power out of battery plant (kW)
    m.bat_e = Var(m.interval, domain=NonNegativeReals)
    m.initial_bat_e = Var(domain=NonNegativeReals)

    #Electricity Connection Point
    m.CP_wind = Var(m.time_periods, domain=NonNegativeReals)
    m.CP_pv= Var(m.time_periods, domain=NonNegativeReals)
    m.CP_el=Var(m.time_periods, domain=NonPositiveReals)
    m.CP_grid_out= Var(m.time_periods, domain=NonPositiveReals)
    m.CP_grid_in= Var(m.time_periods, domain=NonNegativeReals)
    m.CP_comp=Var(m.time_periods, domain=NonPositiveReals)
    m.CP_bat = Var(m.time_periods, domain=Reals)  # can either discharge or charge
    #Compressor node:
    m.comp_pin=Var(m.time_periods, domain=NonNegativeReals)

    #Grid node:
    m.grid_pout= Var(m.time_periods, domain=NonPositiveReals)
    m.grid_pin=Var(m.time_periods, domain=NonNegativeReals)

    #CO2 node:
    m.MEF_CO2 = Var(m.time_periods, domain=Reals)
    m.AEF_CO2 = Var(m.time_periods, domain=Reals)


    # Electrolyser Node:
    m.el_pin = Var(m.time_periods, domain=NonNegativeReals)         #into
    m.el_pout = Var(m.time_periods, domain=NonPositiveReals)        #out

    #H2 connection point node:
    m.H2CP_el= Var(m.time_periods, domain=NonNegativeReals)       #into
    m.H2CP_h2_storage: Var= Var(m.time_periods, domain=Reals)   
    m.H2CP_h2_demand= Var(m.time_periods, domain=NonPositiveReals)       #out

    # H2 storage node:
    m.h2_storage_pin = Var(m.time_periods, domain=NonNegativeReals)     #into
    m.h2_storage_pout = Var(m.time_periods, domain=NonPositiveReals)    #out
    m.h2_storage_level = Var(m.interval, domain=NonNegativeReals)
    m.initial_h2_storage_value = Var(domain=NonNegativeReals)
    m.is_storage_pin_active = Var(m.time_periods, within=Binary)
    m.is_storage_pout_active = Var(m.time_periods, within=Binary)

    '''Constraints'''

    '''Wind and PV generation''' # excess energy will be curtailed
    def constraint_rule_pv(m, i):
        return  m.pv_pout[i]==-1*m.pv_capacity/m.pv_ref_size*m.pv_ref_generation[i]
    m.con_pv= Constraint(m.time_periods, rule=constraint_rule_pv)

    def constraint_rule_wind(m, i):
        return  m.wind_pout[i]==-1*m.wind_capacity/m.wind_ref_size*m.wind_ref_generation[i]
    m.con_wind = Constraint(m.time_periods, rule=constraint_rule_wind)

    '''Electricity connection point'''
    #Grid input and output
    def constraint_rule_CP_grid_1(m, i):
        return  m.grid_pout[i]+m.CP_grid_in[i]==0
    m.con_grid1 = Constraint(m.time_periods, rule=constraint_rule_CP_grid_1)

    def constraint_rule_CP_grid_2(m, i):
        return  m.grid_pin[i]+m.CP_grid_out[i]==0
    m.con_grid2 = Constraint(m.time_periods, rule=constraint_rule_CP_grid_2)

    #PV and Wind input
    def constraint_rule_CP_wind(m, i):
        return  m.wind_pout[i]+m.CP_wind[i]==0
    m.con_CP_wind = Constraint(m.time_periods, rule=constraint_rule_CP_wind)

    def constraint_rule_CP_pv(m, i):
        return  m.pv_pout[i]+m.CP_pv[i]==0
    m.con_CP_pv = Constraint(m.time_periods, rule=constraint_rule_CP_pv)

    # Electrolyser output
    def constraint_rule_CP_el(m, i):
        return  m.el_pin[i]+m.CP_el[i]==0
    m.con_CP_el = Constraint(m.time_periods, rule=constraint_rule_CP_el)

    #Compressor output
    def constraint_rule_CP_comp(m, i):
        return  m.comp_pin[i]+m.CP_comp[i]==0
    m.con_CP_comp = Constraint(m.time_periods, rule=constraint_rule_CP_comp)

    '''Flow balance in electricity connection point'''

    def constraint_rule_CP(m, i):
        return m.CP_el[i] + m.CP_pv[i] + m.CP_wind[i] + m.CP_grid_out[i] + m.CP_grid_in[i] + m.CP_comp[i] + m.CP_bat[i]+m.CP_curtailment[i] == 0
    m.con_CP = Constraint(m.time_periods, rule=constraint_rule_CP)

    '''Compressor'''
    def constraint_rule_comp(m, i):
        return  m.comp_pin[i]== m.comp2_power_conversion*m.h2_storage_pin[i]+0.83*(-m.el_pout[i])
    m.con_comp = Constraint(m.time_periods, rule=constraint_rule_comp)

    '''Grid Node'''

    #Big M method to constraint direction can be two-ways at the same time

    def constraint_rule_CP_grid(m, i):
        return m.grid_pin[i] <= m.is_grid_pin_active[i] * m.M  # M is a large constant
    m.con_grid_pin = Constraint(m.time_periods, rule=constraint_rule_CP_grid)

    def constraint_rule_CP_grid(m, i):
        return -m.grid_pout[i] <= m.is_grid_pout_active[i] * m.M  # M is a large constant
    m.con_grid_pout = Constraint(m.time_periods, rule=constraint_rule_CP_grid)

    def constraint_rule_CP_grid_one_flow(m, i):
        return m.is_grid_pin_active[i] + m.is_grid_pout_active[i] == 1
    m.con_grid_one_flow = Constraint(m.time_periods, rule=constraint_rule_CP_grid_one_flow)

    #sell and buy constraint
    def constraint_rule_grid_buy_max(m, i):
        return  m.grid_pout[i]>=-m.maximum_power_integration
    m.con_grid_buy_max= Constraint(m.time_periods, rule=constraint_rule_grid_buy_max)

    def constraint_rule_grid_sell_max(m, i):
        return  m.grid_pin[i]<=m.maximum_power_integration
    m.con_grid_sell_max= Constraint(m.time_periods, rule=constraint_rule_grid_sell_max)

    #Temporal correlation
    if SO==1:
        if num_interval==0:
            print("Simultaneity_obligation is off")
        else:
            print("Simultaneity_obligation is on")
            if num_interval==720:       #Monthly Simultaneity Obligation (Calendar Month)
                def simultaneity_rule(m, i):
                    return sum(m.grid_pin[t] + m.grid_pout[t] for t in m.time_periods if (t >= cumulative_supply[i]) and (t < cumulative_supply[i+1])) >= 0
                m.simultaneity_constraint = Constraint(range(len(cumulative_supply)-1),rule=simultaneity_rule)
            else:
                def simultaneity_rule(m, j):
                    return sum(m.grid_pin[i] + m.grid_pout[i] for i in range(j, min(j + interval, 8760))) >= 0
                m.simultaneity_constraint = Constraint(m.check_periods, rule=simultaneity_rule)
    if SO==0:
        num_interval=0
        print("Simultaneity_obligation is off")

    '''Electrolyser node'''
    def constraint_rule_el(m, i):      #this indicates the power balance in different time steps for example if step is 15min the power consumed is 1/4 kWh
        return  m.el_pin[i]+m.el_pout[i]*(39.4/0.7)==0
    m.con_el = Constraint(m.time_periods, rule=constraint_rule_el)
    #0.7 is the efficiency of the electrolyser. 39.4 is the conversion rate

    #capacity constraint
    def constraint_rule_el_pin(m, i):
        return  m.el_pin[i]<=m.electrolyser_capacity*step/60
    m.con_el_pin= Constraint(m.time_periods, rule=constraint_rule_el_pin)

    '''H2 connection point'''
    #Flow balance in H2 connection point
    def constraint_rule_H2CP(m, i):
        return  m.H2CP_el[i]+m.H2CP_h2_storage[i]+m.H2CP_h2_demand[i]==0
    m.con_H2CP = Constraint(m.time_periods, rule=constraint_rule_H2CP)

    #input from Electrolyser
    def constraint_rule_H2CP_el(m, i):
        return  m.el_pout[i]+m.H2CP_el[i]==0
    m.con_H2CP_el = Constraint(m.time_periods, rule=constraint_rule_H2CP_el)

    # Output to hydrogen storage
    def constraint_rule_H2CP_H2storage(m, i):
        return  m.H2CP_h2_storage[i]+ m.h2_storage_pin[i]==0
    m.con_H2CP_H2storage= Constraint(m.time_periods, rule=constraint_rule_H2CP_H2storage)

    '''Hydrogen storage Node'''
    #Iteration strategy
    def constraint_rule_H2storage(m, i):
        if i == 0:
            return m.h2_storage_level[i] == m.initial_h2_storage_value+m.h2_storage_pin[i]+m.h2_storage_pout[i]
        else:
            return m.h2_storage_level[i] == m.h2_storage_level[i - 1]+m.h2_storage_pin[i]+m.h2_storage_pout[i]
    m.con_H2storage_level = Constraint(m.time_periods, rule=constraint_rule_H2storage)

    def constraint_rule_H2storage_level(m,i):
        return  m.h2_storage_level[i]>=0
    m.con_H2storage_level1= Constraint(m.time_periods, rule=constraint_rule_H2storage_level)

    #capacity constraint of hydrogen storage
    def constraint_rule_H2storage_capacity(m, i):
        return  m.h2_storage_level[i]<=m.h2_storage_capacity
    m.con_H2storage_capacity= Constraint(m.time_periods, rule=constraint_rule_H2storage_capacity)
    m.con_hydrogen_storage_t = Constraint(expr=m.h2_storage_capacity_t==m.h2_storage_capacity/1000)

    #Hydrogen storage type constraint:
    if hydrogen_storage_type=='Pipeline':
        print('Hydrogen storage type is',hydrogen_storage_type)
        # Piecewise function to calculate the cost
        b, v = piecewise_function(hydrogen_storage_bound, 50, hydrogen_storage_type)  # lower bound is the cross point
        breakpoints = list(b)
        function_points = list(v)
        m.con = Piecewise(m.c_hydrogen_storage, m.h2_storage_capacity_t,
                          pw_pts=breakpoints,
                          pw_constr_type='EQ',
                          f_rule=function_points,
                          pw_repn='INC')
    if hydrogen_storage_type =='Salt Cavern' or hydrogen_storage_type =='Lined Rock':
        print('Hydrogen storage type is',hydrogen_storage_type)
        #Piecewise function to calculate the cost
        b,v=piecewise_function(hydrogen_storage_bound,50,hydrogen_storage_type)  #lower bound is the cross point
        breakpoints =list(b)
        function_points = list(v)
        m.con = Piecewise(m.c_hydrogen_storage, m.h2_storage_capacity_t,
                              pw_pts=breakpoints,
                              pw_constr_type='EQ',
                              f_rule=function_points,
                              pw_repn='INC')

    #Initial level=End level
    m.con_hydrogen_storage = Constraint(expr=m.h2_storage_level[end_index] == m.initial_h2_storage_value)

    '''Battery node'''

    # Output to battery
    def constraint_rule_CP_battery(m, i):
        return m.bat_pin[i] + m.bat_pout[i] + m.CP_bat[i] == 0  # charge and discharge efficiency is 90%

    m.con_CP_battery = Constraint(m.time_periods, rule=constraint_rule_CP_battery)

    # Iteration strategy for battery
    def constraint_rule_battery(m, i):
        if i == 0:
            return m.bat_e[i] == m.initial_bat_e + m.bat_pin[i] * 0.9 + m.bat_pout[i] / 0.9
        else:
            return m.bat_e[i] == m.bat_e[i - 1] + m.bat_pin[i] * 0.9 + m.bat_pout[i] / 0.9

    m.con_battery_level = Constraint(m.time_periods, rule=constraint_rule_battery)

    # Initial level=End level for battery
    m.con_battery = Constraint(expr=m.bat_e[end_index] == m.initial_bat_e)

    def constraint_rule_battery_level(m, i):
        return m.bat_e[i] >= 0
    m.con_battery_level1 = Constraint(m.time_periods, rule=constraint_rule_battery_level)

    # capacity constraint of battery
    def constraint_rule_battery_energy_capacity(m, i):
        return m.bat_e[i] <= m.bat_e_capacity
    m.con_battery_energy_capacity = Constraint(m.time_periods, rule=constraint_rule_battery_energy_capacity)

    def constraint_rule_battery_power_capacity_pos(m, i):
        return m.bat_pin[i] <= m.bat_p_capacity

    m.con_battery_power_capacity_pos = Constraint(m.time_periods, rule=constraint_rule_battery_power_capacity_pos)

    def constraint_rule_battery_power_capacity_neg(m, i):
        return m.bat_pout[i] >= -m.bat_p_capacity
    m.con_battery_power_capacity_neg = Constraint(m.time_periods, rule=constraint_rule_battery_power_capacity_neg)

    '''Load node'''
    m.con_production=Constraint(expr=sum(m.Load[i] for i in m.time_periods)==m.production_amount)

    if batch_interval==720:   #for calendar month
        def load_constraint_rule(m, i):
            return sum(m.Load[t] for t in m.time_periods if (t >= cumulative_supply[i]) and (t < cumulative_supply[i+1])) == hydrogen_load_flow*24*365 /12
        m.load_constraint = Constraint(range(len(cumulative_supply)-1), rule=load_constraint_rule)
    else:
        def load_constraint_rule(m, i):
            return sum(m.Load[t] for t in m.time_periods if (t >= i) and (t < i+batch_interval)) == hydrogen_load_flow*24* 365 / number_of_supply
        m.load_constraint = Constraint(m.supply_periods, rule=load_constraint_rule)

    def constraint_rule_H2CP_H2demand(m, i):
        return  m.H2CP_h2_demand[i]+ m.Load[i]+m.h2_storage_pout[i]==0
    m.con_H2CP_H2demand= Constraint(m.time_periods, rule=constraint_rule_H2CP_H2demand)

    '''Indicators'''
    #Grid interaction cost
    m.con_cost=Constraint(expr=m.grid_electricity_cost==sum(-m.grid_pout[i]*(m.price[i]+0.01)-(m.grid_pin[i]*(m.price[i])) for i in m.time_periods))
    #0.034 means the integration cost of equipment which can help newly installed renewable energy reliable
    #0.01 is TUOS

    #CO2
    def constraint_rule_CO2(m, i):
        return  m.MEF_CO2[i]==-(m.grid_pout[i]+m.grid_pin[i])*m.MEF[i]      #According to Energy interaction
    m.con_CO2= Constraint(m.time_periods, rule=constraint_rule_CO2)

    def constraint_rule_Mean_CO2(m, i):
        return  m.AEF_CO2[i]==-(m.grid_pout[i]+m.grid_pin[i])*m.AEF[i]
    m.con_Mean_CO2= Constraint(m.time_periods, rule=constraint_rule_Mean_CO2)

    #Carbon Emission Requirement
    m.con_carbon_emission=Constraint(expr=sum(-1*m.grid_pout[i]*(1-0.188)-m.grid_pin[i] for i in m.time_periods)<=0)
    #m.con_carbon_emission=Constraint(expr=sum(m.MEF_CO2[i] for i in m.time_periods)<=0)


    # LCOH and capex
    m.con_capex = Constraint(expr=m.capex == m.c_pv * m.pv_capacity +
                              m.c_wind * m.wind_capacity +
                              m.c_el * m.electrolyser_capacity +
                              m.c_hydrogen_storage * m.h2_storage_capacity +
                              m.c_bat_e * m.bat_e_capacity + m.c_bat_p * m.bat_p_capacity)


    def LCOH_constraint(m,i):
        return m.LCOH == (
            (m.capex * m.CRF + m.pv_capacity * m.pv_FOM + m.wind_capacity * m.wind_FOM + m.electrolyser_capacity* m.el_FOM+m.grid_electricity_cost*0.7))
            # /m.production_amount
            # + m.el_VOM)
    m.LCOH_constraint = Constraint(rule=LCOH_constraint)
    #0.7 is the ratio between USD:AUD

    m.obj = Objective(expr=m.LCOH, sense=minimize)
    # Solve the linear programming problem
    solver = SolverFactory('gurobi')              #'Cplex', 'ipopt'
    solver.options['NonConvex'] = 2
    #solver.options['Tol'] = 1e-5
    results = solver.solve(m)

    '''Result printout'''
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal Solution Found")
        variable_values = {}
        # print the flow value
        CP_grid_out = list()
        CP_grid_in = list()
        grid_interaction = list()
        grid_pout = list()
        grid_pin = list()
        CP_wind = list()
        CP_pv = list()
        CP_el = list()
        CP_bat = list()
        pv_pout = list()
        wind_pout = list()
        curtailment= list()
        el_pin = list()
        el_pout = list()
        comp_pin = list()
        bat_pin=list()
        bat_pout = list()
        bat_e=list()
        h2_storage_pin = list()
        h2_storage_pout = list()
        H2CP_h2_storage=list()
        h2_storage_level = list()
        price = list()
        MEF_CO2 = list()
        AEF_CO2 = list()
        load = list()

        for Time in m.time_periods:
            CP_grid_out.append(value(m.CP_grid_out[Time]))
            CP_grid_in.append(value(m.CP_grid_in[Time]))
            grid_interaction.append(value(m.grid_pout[Time]) + value(m.grid_pin[Time]))
            grid_pout.append(value(m.grid_pout[Time]))
            grid_pin.append(value(m.grid_pin[Time]))
            pv_pout.append(value(m.pv_pout[Time]))
            wind_pout.append(value(m.wind_pout[Time]))
            curtailment.append(value(m.CP_curtailment[Time]))
            CP_wind.append(value(m.CP_wind[Time]))
            CP_pv.append(value(m.CP_pv[Time]))
            CP_el.append(value(m.CP_el[Time]))
            CP_bat.append(value(m.CP_bat[Time]))
            el_pin.append(value(m.el_pin[Time]))
            el_pout.append(value(m.el_pout[Time]))
            comp_pin.append(value(m.comp_pin[Time]))
            bat_pin.append(value(m.bat_pin[Time]))
            bat_pout.append(value(m.bat_pout[Time]))
            bat_e.append(value(m.bat_e[Time]))
            h2_storage_pin.append(value(m.h2_storage_pin[Time]))
            h2_storage_pout.append(value(m.h2_storage_pout[Time]))
            H2CP_h2_storage.append(value(m.H2CP_h2_storage[Time]))
            h2_storage_level.append(value(m.h2_storage_level[Time]))
            price.append(value(m.price[Time]))
            MEF_CO2.append(value(m.MEF_CO2[Time]))
            AEF_CO2.append(value(m.AEF_CO2[Time]))
            load.append(value(m.Load[Time]))

        data = {'grid_interaction': grid_interaction,
                'CP_grid_out': CP_grid_out,
                'CP_grid_in': CP_grid_in,
                'grid_pout': grid_pout,
                'grid_pin': grid_pin,
                'pv_pout': pv_pout,
                'wind_pout': wind_pout,
                'curtailment': curtailment,
                'CP_wind': CP_wind,
                'CP_pv': CP_pv,
                'CP_el': CP_el,
                'CP_bat':CP_bat,
                'el_pin': el_pin,
                'el_pout': el_pout,
                'comp_pin': comp_pin,
                'bat_pin':bat_pin,
                'bat_pout':bat_pout,
                'bat_e':bat_e,
                'h2_storage_pin': h2_storage_pin,
                'h2_storage_pout': h2_storage_pout,
                'H2CP_h2_storage':H2CP_h2_storage,
                'h2_storage_level': h2_storage_level,
                'price': price,
                'MEF_CO2': MEF_CO2,
                'AEF_CO2': AEF_CO2,
                'Load': load,

                }
        # Create a DataFrame
        df = pd.DataFrame(data)
        if hydrogen_storage_type == 'Lined Rock':
            df['Direct_pipeline_supply'] = df['Load']+df['h2_storage_pout']
        else: #pipeline
            df['Direct_pipeline_supply'] = np.where(df['h2_storage_pin']+df['h2_storage_pout']>=0, df['Load'], df['Load']+(df['h2_storage_pin']+df['h2_storage_pout']))

        df['Time'] = Spotprice(year, 'QLD1', 60)['Time']

        #Test the flow direction
        '''
        if hydrogen_storage_type == 'Lined Rock':
            if (df['h2_storage_pout']*df['h2_storage_pin']).sum()==0:
                print('Hydrogen Storage Interaction Flow is Correct')

            else:
                print('Hydrogen Storage Interaction Flow is Incorrect')
                sys.exit()

        '''
        if (df['grid_pout']*df['grid_pin']).sum()==0:
            print('Grid Flow Interaction is Correct')
        else:
            print('Grid Flow Interaction is Incorrect')
            sys.exit()



        '''Save key indicators:'''
        production_amount = sum(df['Load'])
        purchase_amount = sum(df['grid_pout'])
        sell_amount = sum(df['grid_pin'])
        Curtailment=sum(df['curtailment'])
        RE_generation = sum(df['CP_wind'] + df['CP_pv'])+Curtailment
        Supply_proportion = (sum(df['el_pin'] + df['comp_pin']) - sell_amount) / sum(df['el_pin'] + df['comp_pin'])  # which shows the power supply comes from which source
        LCOH = (value(m.LCOH)) / production_amount + value(m.el_VOM)


        ''' location-based carbon emission calculation method:  (kg CO2e/kWh)'''
        if location == 'QLD1':
            EI_location_based_method = -sum(df['grid_interaction'] * (0.73)) / production_amount
        if location == 'NSW1':
            EI_location_based_method = -sum(df['grid_interaction'] * (0.68)) / production_amount
        if location == 'TAS1':
            EI_location_based_method = -sum(df['grid_interaction'] * (0.12)) / production_amount
        if location == 'VIC1':
            EI_location_based_method = -sum(df['grid_interaction'] * (0.79)) / production_amount
        if location == 'SA1':
            EI_location_based_method = -sum(df['grid_interaction'] * (0.25)) / production_amount

        ''' Market-based carbon emission calculation method:  (kg CO2e/kWh)'''
        market_based_emission = (-1 * purchase_amount * (1 - 0.188) - (sell_amount)) * 0.81 / production_amount
        LGCs = (-1 * purchase_amount * (1 - 0.188) - (sell_amount)) / 1000

        capex = value(m.capex)
        grid_electricity_cost = value(m.grid_electricity_cost)
        production_amount = sum(df['Load'])
        MEF_carbon_emissions = sum(df['MEF_CO2']) / value(m.production_amount)
        AEF_carbon_emissions = sum(df['AEF_CO2']) / value(m.production_amount)
        h2_initial_storage_level = value(m.initial_h2_storage_value)
        battery_energy_capacity=value(m.bat_e_capacity)
        battery_power_capacity = value(m.bat_p_capacity)
        wind_capacity = value(m.wind_capacity)
        pv_capacity = value(m.pv_capacity)
        electrolyser_capacity = value(m.electrolyser_capacity)
        hydrogen_storage_capacity = value(m.h2_storage_capacity)
        maximum_power_integration = value(m.maximum_power_integration)
        Full_load_hours = sum(df['el_pin']) / value(m.electrolyser_capacity)

        print("Simultaneity_obligation_interval:", num_interval)
        print("LCOH:", LCOH)
        print("Capex:", capex)
        print('Model running resolution:', step)
        print("Average_grid_electricity_cost:", value(m.grid_electricity_cost) / value(m.production_amount))
        print("production_amount:", sum(df['Load']))
        print("EI_MEF:", sum(df['MEF_CO2']) / value(m.production_amount))
        print("EI_AEF:", sum(df['AEF_CO2']) / value(m.production_amount))
        print("EI_L:", EI_location_based_method)
        print("EI_market:", market_based_emission)
        print("H2_initial_storage_level:", value(m.initial_h2_storage_value),
              "\nH2_final_storage_level:", value(m.h2_storage_level[end_index]))
        print("Battery_initial_level:", value(m.initial_bat_e),
              "\nBattery_final_level:", value(m.bat_e[end_index]))
        print("Unit Capex of Hydrogen storage is", value(m.c_hydrogen_storage))
        print("Battery type:", c_bat_class)
        print("c_bat_e:",value(m.c_bat_e))
        print("c_bat_p:",value(m.c_bat_p))

        # Create a dictionary with key indicators
        indicators = {
            'Location': location,
            'Location_code':location_code,
            'Year': year,
            'LCOH': LCOH,
            'Capex': capex,
            'Step': step,
            'RE_supply_proportion': Supply_proportion,
            'FLH': Full_load_hours,
            'EI_market': market_based_emission,
            'EI_MEF': MEF_carbon_emissions,
            'EI_AEF': AEF_carbon_emissions,
            'EI_L': EI_location_based_method,
            'Sell_amount': sell_amount,
            'Purchase_amount': purchase_amount,
            'Curtailment': Curtailment,
            'LGCs': LGCs,
            'battery class':c_bat_class,
            'battery_energy_capacity':battery_energy_capacity,
            'battery_power_capacity':battery_power_capacity,
            'wind_capacity': wind_capacity,
            'pv_capacity': pv_capacity,
            'electrolyser_capacity': electrolyser_capacity,
            'hydrogen_storage_capacity': hydrogen_storage_capacity,
            'hydrogen_storage_type': hydrogen_storage_type,
            'hydrogen_storage_cost':value(m.c_hydrogen_storage),
            'Grid_max_power_export': maximum_power_integration,
            'grid_cost': grid_electricity_cost,
            'production_amount': production_amount,
            'H2_initial_storage_level': h2_initial_storage_level,
            'Simultaneity_obligation_interval': num_interval,
            'ratio': ratio,
            'Simultaneity_obligation': SO,
            'Batch_interval': batch_interval,
        }

        # Create a DataFrame
        indicators_df = pd.DataFrame([indicators])
        return df, indicators_df
    else:
        print("No optimal solution found")
        print("Solver Status:", results.solver.status)

        return None, None







    


































































