import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

csv_file_path = 'non_renorm_out.dat'
csv_file_path_2 = 'renorm_out.dat'
csv_file_path_2 = 'output.dat'

# Read only the first 13 columns to avoid extra unnamed columns
df = pd.read_csv(csv_file_path, sep='\s+', usecols=range(13))
df_2 = pd.read_csv(csv_file_path_2, sep='\s+', usecols=range(13))

var1 = 'k'
type1 = float
type2 = float

# Get the list of column names
column_names = df.columns.tolist()
num_plots = len(column_names) - 1  # Exclude the first column (x-axis)

# Debugging: Check the columns and axes length
print(f"Columns to plot (excluding x-axis): {column_names[1:]}")
print(f"Total number of plots: {num_plots}")

# Calculate the number of rows and columns for subplots
ncols = 4  # Number of columns in subplot grid
nrows = math.ceil(num_plots / ncols)

# Set up the grid of subplots
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
axes = axes.flatten()

# Loop over the column names, skipping the first one (assuming it's 'k')
for idx, column_name in enumerate(column_names[1:]):
    if idx >= len(axes):
        print(f"Warning: More columns to plot ({num_plots}) than subplots available ({len(axes)}).")
        break

    k1 = df.iloc[:, 0].astype(type1)
    p_cdm1 = df[column_name].astype(type2)

    k2 = df_2.iloc[:, 0].astype(type1)
    p_cdm2 = df_2[column_name].astype(type2)

    # Interpolate p_cdm2 to match k1
    interp_p_cdm2 = np.interp(k1, k2, p_cdm2)
    ratio = p_cdm1 / interp_p_cdm2

    # Select the current subplot
    ax = axes[idx]

    ax.plot(k1, p_cdm1, label='Working PS')
    ax.plot(k2, p_cdm2, label='Broken PS')
    # Uncomment to plot the ratio
    # ax.plot(k1, ratio, label=f'Ratio, {column_name}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(column_name)

# Hide any unused subplots
for ax in axes[num_plots:]:
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.suptitle('Comparison of working and broken CAMB power spectra', y=1.02)
plt.subplots_adjust(top=0.9)  # Adjust space for the suptitle
plt.show()
