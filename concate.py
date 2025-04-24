import os
import pandas as pd

# Specify the folder containing CSV files
folder_path = "C:/Users/Kevan/PhD work/Pycharm project/OpenElectricity/Optimization model/Result/on grid/EI_market"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read and concatenate all CSV files (with index_col=0 to use the first column as the index)
df_list = [pd.read_csv(os.path.join(folder_path, file), index_col=0) for file in csv_files]

# Combine all DataFrames into one
final_df = pd.concat(df_list, ignore_index=True)
# Display the combined DataFrame
final_df.to_csv('geographic correlation with EI_market constraints.csv')
