import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file_path = '../../music/camb.dat'
csv_file_path_2 = 'output.dat'
#csv_file_path_2 = '../../pp-music-interface/camb.dat'

#csv_file_path = '../interface_music_run1/input_powerspec.txt'
##csv_file_path_2 = '../interface_music_run1/input_powerspec2.txt'
#csv_file_path_2 = '../../pp-music-interface/input_powerspec.txt'
df = pd.read_csv(csv_file_path, sep='\s+')
df_2 = pd.read_csv(csv_file_path_2, sep='\s+')

var1 = 'k'
var2 = 1
type1 = float
type2 = float

# Set up a 3x4 grid of subplots (for 12 columns)
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Flatten axes to easily iterate over them
axes = axes.flatten()

for column in range(1, 12):
    
    k1 = df.iloc[:, 0].astype(type1)
    p_cdm1 = df.iloc[:, column].astype(type2)
    
    k2 = df_2.iloc[:, 0].astype(type1)
    p_cdm2 = df_2.iloc[:, column].astype(type2)

    interp_p_cdm2 = np.interp(k1, k2, p_cdm2)
    ratio = p_cdm1 / interp_p_cdm2
    
    # Select the current subplot
    ax = axes[column-1]
    
    ax.plot(k1, p_cdm1, label=f'Working PS')
    ax.plot(k2, p_cdm2, label=f'Broken PS')
    # Uncomment to plot the ratio
    # ax.plot(k1, ratio, label=f'Ratio, column {column}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(f'Column {column}')
    
# Adjust layout
plt.tight_layout()
plt.suptitle('Comparison of working and broken CAMB power spectra')
plt.subplots_adjust(top=0.9)  # Adjust space for the suptitle
plt.show()
