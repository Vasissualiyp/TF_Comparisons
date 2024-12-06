import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# File paths
figure_name = 'CLASS_vs_CAMB_full_comparison.png'

csv_file_path_1 = 'CLASS.dat'
csv_file_path_2 = 'CAMB.dat'
label_1 = 'CLASS'
label_2 = 'CAMB'

csv_files = [ csv_file_path_1, csv_file_path_2 ]
labels    = [ label_1,         label_2         ]

ratio_plot = False
skiprows = 0 # 1 for data with the header (most), 0 for data w/o header (PeakPatch)


#----------------------------------------------------------------------------
#-------------------------- END OF MODIFYABLE PART -------------------------- 
#----------------------------------------------------------------------------

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

def get_rows_columns(num_plots):
    """
    Get a number of rows and columns for the subplots of the main plot, given the total
    number of subplots

    Args:
        num_plots (int): number of subplots

    Returns:
        nrows (int): number of rows 
        ncols (int): number of columns 
    """
    if num_plots < 5:
        ncols = num_plots
    else:
        ncols = math.ceil(np.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)
    return nrows, ncols

def get_figsize(nrows, ncols, plot_size, bnd_size):
    """
    Gets the geometric size of the figure based on subfigure structure

    Args:
        nrows (int): number of rows of subfigures
        ncols (int): number of columns of subfigures
        plot_size (float): size of the figure
        bnd_size (float): size of the  boundary

    Returns:
        hsize (float): horizontal size of master plot
        vsize (float): vertical size of master plot
    """
    hsize = ncols * plot_size + 2 * bnd_size
    vsize = nrows * plot_size + 2 * bnd_size
    return hsize, vsize

def plot_single_TF(ax, k, p_cdm, label):
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

def plot_TF_ratio(ax, dfs, x_column_name, column_name):
    """
    Plots a comparison plot of a single Transfer Function (TF) type.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to plot on.
        dfs (list of pd dfs): list of datasets.
        x_column_name (str): Name of the column with k's.
        column_name (str): Name of the TF type.

    Returns:
        None
    """
    k1 = np.array(dfs[0][x_column_name]).flatten() 
    k2 = np.array(dfs[1][x_column_name]).flatten() 
    p_cdm1 = np.array(dfs[0][column_name]).flatten() 
    p_cdm2 = np.array(dfs[1][column_name]).flatten() 
    # Interpolate p_cdm2 to match k1
    interp_p_cdm2 = np.interp(k1, k2, p_cdm2)
    ratio = p_cdm1 / interp_p_cdm2
    ax.plot(k1, ratio, label=f'Ratio, {column_name}')

def plot_single_TF_comparison(dfs, ax, x_column_name, column_name, labels, ratio_plot=False):
    """
    Plot comparison of single type of TFs

    Args:
        dfs (list of pd dfs): list of datasets.
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot axis to plot on.
        idx (int): id of the TF
        x_column_name (str): Name of the column with k's.
        column_name (str): Name of the TF type.
        labels (list of str): List of labels for each of the datasets.
        ratio_plot (bool): Whether to do a plot of ratios, or just TFs.

    Returns:
        None
    """
    # Plot the data
    if ratio_plot:
        if len(dfs) == 2:
            plot_TF_ratio(ax, dfs, x_column_name, column_name)
        else:
            print(f"Plotting ratio is only possible if you passed 2 datasets. You passed {len(dfs)}")
            print("Please, pass less datasets or set ratio_plot to False")
            exit(1)
    else:
        print(len(dfs))
        for df, label in zip(dfs, labels):
            plot_single_TF(ax, df[x_column_name], df[column_name], label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(column_name)

def obtain_headers(csv_file_path, skiprows):
    """
    Get the header column names from the file or pre-defined for PeakPatch

    Args:
        csv_file_path (str): path to the csv/dat table
        skiprows (int): how many rows to skip (usually 1 for header or 0 for header-less)

    Returns:
        header_columns (list): list of column names
    """
    pp_header = 'k, Pk, Trans, pkchi'
    if skiprows == 0:
        header_line = set_header(csv_file_path, pp_header)
    elif skiprows == 1:
        header_line = set_header(csv_file_path)
    else:
        print(f"Wrong value of skiprows: {skiprows}. Allowed values: 0 for headless, 1 for dataset with the header")
        exit(1)
    
    # Process the header to remove the '#' and split by comma
    header_line = header_line.lstrip('#').replace(" ", "")  # Remove leading '#' and spaces
    header_columns = header_line.split(',')  # Split the header by comma
    return header_columns

def setup_plotting(num_plots):
    """
    Generate subplot grid from the number of plots

    Args:
        num_plots (int): number of subplots

    Returns:
        fig (matplotlib.figure.Figure): the master figure
        axes (np array of matplotlib.axes._subplots.AxesSubplot): array of subplots
    """
    nrows, ncols = get_rows_columns(num_plots)
    hsize, vsize = get_figsize(nrows, ncols, 4, 2)
    
    # Set up the grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(hsize, vsize))
    axes = axes.flatten()
    return fig, axes

def finalize_plotting(axes, num_plots, figure_name):
    """
    Finalize plot and save it

    Args:
        axes (np array of matplotlib.axes._subplots.AxesSubplot): array of subplots
        num_plots (int): number of subplots.
        figure_name (str): the name of output plot.

    Returns:
        None
    """
    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Comparison of working and broken CAMB power spectra', y=1.02)
    plt.subplots_adjust(top=0.9)  # Adjust space for the suptitle
    plt.savefig(figure_name)
    plt.show()

def main(skiprows, csv_file_paths, labels, figure_name, ratio_plot=False):
    """
    Main loop for comparison of the plots

    Args:
        skiprows (int): How many rows to skip. Possible values: 0 for headless, 1 for dataset with the header.
        csv_file_paths (list of str): List of path to the file with the datasets.
        labels (list of str): List of labels for each of the datasets.
        figure_name (str): the name of output plot.
        ratio_plot (bool): Whether to do a plot of ratios, or just TFs.

    Returns:
        int: 0 if success
    """

    header_columns = obtain_headers(csv_file_paths[0], skiprows)
    
    dfs = []
    for csv_file_path in csv_file_paths:
        dfs.append( read_data(csv_file_path, skiprows, header_columns) )
    
    # Get the list of column names, excluding the first one (x-axis)
    x_column_name  = dfs[0].columns[0]  # This should now be 'k'
    y_column_names = dfs[0].columns[1:]  # All columns excluding 'k'
    
    num_plots = len(y_column_names)
    
    # Set up the grid of subplots
    fig, axes = setup_plotting(num_plots)
    
    # Loop over the column names
    for idx, column_name in enumerate(y_column_names):
        # Select the current subplot
        ax = axes[idx]
        plot_single_TF_comparison(dfs, ax, x_column_name, column_name, labels, ratio_plot)
    
    finalize_plotting(axes, num_plots, figure_name)

    return 0

main( skiprows, csv_files, labels, figure_name, ratio_plot ) 
