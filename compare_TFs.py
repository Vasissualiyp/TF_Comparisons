import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# File paths
#csv_file_path = 'non_renorm_out.dat'
#csv_file_path_2 = 'output.dat'
csv_file_path = 'CLASS.dat'
label_1 = 'CLASS'
csv_file_path_2 = 'CAMB.dat'
label_2 = 'CAMB'
figure_name = 'CLASS_vs_CAMB_full_comparison.png'

# Read the header separately, assuming the first line is the header
with open(csv_file_path, 'r') as f:
    header_line = f.readline().strip()

# Process the header to remove the '#' and split by comma
header_line = header_line.lstrip('#').replace(" ", "")  # Remove leading '#' and spaces
header_columns = header_line.split(',')  # Split the header by comma

# Read the data, skipping the header line, using spaces as delimiters
df = pd.read_csv(csv_file_path, sep='\s+', skiprows=1, names=header_columns, usecols=range(13))
df_2 = pd.read_csv(csv_file_path_2, sep='\s+', skiprows=1, names=header_columns, usecols=range(13))

# Ensure the type of data is correct
df = df.astype(float)
df_2 = df_2.astype(float)

# Get the list of column names, excluding the first one (x-axis)
x_column_name = df.columns[0]  # This should now be 'k'
y_column_names = df.columns[1:]  # All columns excluding 'k'

num_plots = len(y_column_names)

# Debugging: Check the columns and axes length
print(f"x-axis column: {x_column_name}")
print(f"Columns to plot (excluding x-axis): {y_column_names}")
print(f"Total number of plots: {num_plots}")

# Calculate the number of rows and columns for subplots
ncols = 4  # Number of columns in subplot grid
nrows = math.ceil(num_plots / ncols)

# Set up the grid of subplots
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
axes = axes.flatten()

# Loop over the column names
for idx, column_name in enumerate(y_column_names):
    k1 = df[x_column_name]  # First column is always the x-axis (k)
    p_cdm1 = df[column_name]  # Extract the current column for plotting

    k2 = df_2[x_column_name]  # First column for the second dataset (k)
    p_cdm2 = df_2[column_name]  # Extract the corresponding column from the second dataset

    # Interpolate p_cdm2 to match k1
    interp_p_cdm2 = np.interp(k1, k2, p_cdm2)
    ratio = p_cdm1 / interp_p_cdm2

    # Select the current subplot
    ax = axes[idx]

    # Plot the data
    ax.plot(k1, p_cdm1, label=label_1)
    ax.plot(k2, p_cdm2, label=label_2)
    # Uncomment to plot the ratio if needed
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
plt.savefig(figure_name)
