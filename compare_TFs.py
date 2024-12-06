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

header = 'k, Pk, Trans, pkchi'
skiprows = 0 # 1 for data with the header (most), 0 for data w/o header (PeakPatch)

def set_header(csv_file_path, header=None):
    """
    This function returns header name if it is passed, or reads it from the file if it's not passed

    Args:
        csv_file_path (str): path to the csv/dat table
        header (str): header, which is column names, separated by comma. 
            Set to nothing to read it from the file.

    Returns:
        header (str): header
    """
    if header == None:
        with open(csv_file_path, 'r') as f:
            header_line = f.readline().strip()
        return header_line
    else:
        return header


if skiprows == 0:
    header_line = set_header(csv_file_path, header)
elif skiprows == 1:
    header_line = set_header(csv_file_path)
# Process the header to remove the '#' and split by comma
header_line = header_line.lstrip('#').replace(" ", "")  # Remove leading '#' and spaces
header_columns = header_line.split(',')  # Split the header by comma

# Read the data, skipping the header line, using spaces as delimiters
def read_data(csv_file_path, skiprows, header_columns):
    """
    Reads the csv/dat file into dataframe

    Args:
        csv_file_path (str): path to the csv/dat table
        skiprows (int): how many rows to skip (usually 1 for header or 0 for header-less)
        header_columns (list): list of column names

    Returns:
        df (pd dataframe): pandas dataframe, containing read data
    """
    column_no = len(header_columns)
    delim = '\s+' # Spaces as delimiters
    df = pd.read_csv(csv_file_path, sep=delim, skiprows=skiprows, names=header_columns, usecols=range(column_no))
    # Ensure the type of data is correct
    df = df.astype(float)
    return df

df_1 = read_data(csv_file_path, skiprows, header_columns)
df_2 = read_data(csv_file_path_2, skiprows, header_columns)

# Get the list of column names, excluding the first one (x-axis)
x_column_name = df_1.columns[0]  # This should now be 'k'
y_column_names = df_1.columns[1:]  # All columns excluding 'k'

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

def plot_single_TF_comparison(ax, k, p_cdm, label):
    """
    Plots a comparison plot of a single Transfer Function (TF) type.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to plot on.
        k (pandas.Series or numpy.ndarray): List or array of k-values (x-axis data).
        p_cdm (pandas.Series or numpy.ndarray): List or array of TF values (y-axis data).
        label (str): Label for the plot legend.

    Returns:
        None
    """
    ax.plot(k, p_cdm, label=label)

# Loop over the column names
for idx, column_name in enumerate(y_column_names):
    k1 = df_1[x_column_name]  # First column is always the x-axis (k)
    p_cdm1 = df_1[column_name]  # Extract the current column for plotting

    k2 = df_2[x_column_name]  # First column for the second dataset (k)
    p_cdm2 = df_2[column_name]  # Extract the corresponding column from the second dataset

    # Interpolate p_cdm2 to match k1
    interp_p_cdm2 = np.interp(k1, k2, p_cdm2)
    ratio = p_cdm1 / interp_p_cdm2

    # Select the current subplot
    ax = axes[idx]

    # Plot the data
    plot_single_TF_comparison(ax, k1, p_cdm1, label_1)
    plot_single_TF_comparison(ax, k2, p_cdm2, label_2)
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
plt.savefig(figure_name)
plt.show()
