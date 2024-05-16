'''initial value of cost, unit: USD'''
'''HILTCRC data'''
m.c_pv = Param(initialize=1122.7)  # CAPEX of pv
m.c_wind = Param(initialize=1455)  # CAPEX of wind
m.c_el = Param(initialize=1067)  # CAPEX of electrolyser
m.c_hydrogen_storage = Param(initialize=hydrogen_storage_cost)  # CAPEX of hydrogen underground storage (salt cavern) 17.66
m.CRF = Param(initialize=0.07822671821)
m.pv_FOM = Param(initialize=12.7)
m.wind_FOM = Param(initialize=18.65)
m.el_FOM = Param(initialize=37.4)
m.el_VOM = Param(initialize=0.075)  # water

'''Update cost in 2023'''
m.c_pv = Param(initialize=1068.2)  # CAPEX of pv
m.c_wind = Param(initialize=2126.6)  # CAPEX of wind
m.c_el = Param(initialize=1343.3)  # CAPEX of electrolyser

m.CRF = Param(initialize=0.07822671821)
m.pv_FOM = Param(initialize=11.9)
m.wind_FOM = Param(initialize=17.5)
m.el_FOM = Param(initialize=37.4)
m.el_VOM = Param(initialize=0.02)  # the cost of water consumption
#m.c_hydrogen_storage = Param(initialize=hydrogen_storage_cost)  # CAPEX of hydrogen underground storage (salt cavern) 17.66