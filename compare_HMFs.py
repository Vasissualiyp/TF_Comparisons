usr_path = '/home/vasilii/research/software_src/'
usr_path = '../../peakpatch/python/'
import os, sys, math, numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
sys.path.insert(0, usr_path+'peakpatch/python')
from peakpatchtools import PeakPatch

usr_path_vas = '/home/vasilii/research/sims/PeakPatch/'
usr_path_cita = '/fs/lustre/scratch/vpustovoit/PeakPatch/'
out_path_cita =  '/cita/d/www/home/vpustovoit/plots'

machine = 'cita'


if machine == 'cita':
    total_usr_path = usr_path_cita
    sims_dir = usr_path_cita
    out_dir = out_path_cita
elif machine == 'vas':
    total_usr_path = usr_path_vas
    sims_dir = usr_path2
    #out_dir = "/home/vasilii/research/notes/2024/10/03/figures"
    out_dir = "."
else:
    print(f"Unknown machine: {machine}. Allowed values: cita, vas")
    exit(1)

def makepath(date, redshift, runno):
    """
    Takes in date in YYYY-MM-DD format and redshift, returns path to PeakPatch run, 
    as well as its label
    """
    year, month, day = date.split("-")
    runs_collection_dirname = date + "_highz_tests" + str(runno)
    runs_dir = os.path.join(total_usr_path, 'pp_runs', year, month, runs_collection_dirname)

    redshift = str(redshift)
    redshift_dir = "z" + redshift
    single_redshift_run_dir = os.path.join(runs_dir, redshift_dir)
    single_redshift_run_dir += "/"

    label = "z=" + redshift
    return single_redshift_run_dir, label

def make_paths_labels(date, redshifts_list, runno):
    """
    Takes in date in YYYY-MM-DD format and redshift, returns list of paths to PeakPatch run, 
    as well as their labels list
    """
    paths = []
    labels = []
    for redshift in redshifts_list:
        path, label = makepath(date, redshift, runno)
        paths.append(path)
        labels.append(label)
    return paths, labels

#run2_label = 'PeakPatch (IPR)'

# ------------------ PARTS CHANGING BEGIN ----------------------
#run1_path = total_usr_path + 'pp_runs/hpkvd-interface-run3/'
#run2_path = total_usr_path + 'pp_runs/nong_test_z5/'
#run3_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z8/'
#run4_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z11/'
#run5_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z13/'
#
#run1_label = "hpkvd run z=0"
#run2_label = "hpkvd run z=5"
#run3_label = "hpkvd run z=8"
#run4_label = "hpkvd run z=11"
#run5_label = "hpkvd run z=13"

#run_paths =  [ run1_path,  run2_path,  run3_path,  run4_path,  run5_path  ]
#run_labels = [ run1_label, run2_label, run3_label, run4_label, run5_label ]

date = "2025-03-05"
runnos_list =  [1,  2,  3, 4, 5]
boxsize_list = [20, 10, 5, 2, 1]
redshifts_lists = [ 
                    [10, 13, 15, 17],
                    [10, 13, 15, 17],
                    [10, 13, 15, 17],
                    [10, 13, 15, 17],
                    [10, 13, 15, 17]
                    #[10, 13, 15, 17, 20, 23], 
                    #[10, 13, 15, 17, 20, 23, 26, 30]
                  ]
#date = "2025-03-04"
#runnos_list = [2, 3, 4]
#boxsize_list = [200, 100, 50]
#redshifts_lists = [ 
#                    [10, 13],
#                    [10, 13, 15],
#                    [10, 13, 15, 17]
#                    #[10, 13, 15, 17, 20, 23], 
#                    #[10, 13, 15, 17, 20, 23, 26, 30]
#                  ]

# ------------------ PARTS CHANGING END ------------------------

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

class compare_HMFs():
    """A class for plotting the HMF/overdensity comparisons"""
    def __init__(self, run_paths, run_labels, out_dir):
        """Declare classes"""
        if len(run_paths) != len(run_labels):
            print("ERROR: Different lengths of run_paths and run_labels lists!")
            return 1
        self.runs_num=len(run_paths)
        self.runs = []
        self.hist_runs = []
        self.bin_edges_runs = []
        self.halo_hist_runs = []
        self.xedges_runs = []
        self.yedges_runs = []
        self.hist2d_runs = []
        self.runs_psx = []
        self.runs_psy = []
        self.plots_colors=[ 'red', 'blue', 'green', 'orange', 'pink', 'aqua', 'purple' ]
        
        self.run_paths = run_paths
        self.out_dir = out_dir
        self.run_labels = run_labels

    def compute_hmf(self, run, i, cutoff_factor=4, hmf_type='dndogm'):
            hist_run, bin_edges_run = run.hmf(hmf_type=hmf_type)
            # Set up cutoff to avoid plotting "step-like" data at low-m limit
            size_limit = cutoff_factor * run.cellsize * u.Mpc
            hmf_x_left_lim = run.rho_crit * 4/3 * np.pi * size_limit**3 / u.solMass
            # Filter out values to the right of cutoff
            mask = bin_edges_run[1:] >= hmf_x_left_lim
            filtered_bin_edges_run = bin_edges_run[1:][mask]
            filtered_hist_run = hist_run[mask]
            # Append the data
            self.hist_runs.append(filtered_hist_run)
            self.bin_edges_runs.append(filtered_bin_edges_run)
            print(f"Run {i}: HMF computed")

    def compute_density_field_and_ps(self, run, i):
        run.add_field(field_type='rhog')
        run.get_power_spectrum(field_type='rhog', overwrite=(self.run_labels[i] == "hpkvd run z=11"))
        self.runs_psx.append(run.k_from_rhog)
        self.runs_psy.append(run.k_from_rhog**3 / (2 * np.pi**2) * run.p_from_rhog)

    def compute_2d_halo_hist(self, run, i):
        halo_hist_run, xedges_run, yedges_run = run.halo_hist2d()
        self.halo_hist_runs.append(halo_hist_run)
        self.xedges_runs.append(xedges_run)
        self.yedges_runs.append(yedges_run)
        print(f"Run {i}: 2D halo histogram computed")

    def plot_runs(self, out_name, hmf_type='dndlogm', hmf_colspan=2):
        # Initialize lists to store run data
        self.hist_runs = []
        self.bin_edges_runs = []
        self.halo_hist_runs = []
        self.xedges_runs = []
        self.yedges_runs = []
        self.runs_psx = []
        self.runs_psy = []

        ps_colspan = hmf_colspan

        # Process each run
        for i in range(self.runs_num):
            # Load run data
            run = PeakPatch(run_dir=self.run_paths[i], params_file=self.run_paths[i] + 'param/parameters.ini')
            run.add_halos()
            print(f"Run {i}: Halos added")
            self.compute_hmf(run, i, cutoff_factor=4, hmf_type=hmf_type)
            self.compute_density_field_and_ps(run, i)
            self.compute_2d_halo_hist(run, i)
            self.runs.append(run)

        # Create figure with adjusted layout
        nrows = 2
        ncols = max(hmf_colspan, ps_colspan) + self.runs_num  # Total columns in the grid

        fig = plt.figure(figsize=(4 * ncols, 3 * nrows))  # Adjust size as needed

        # Create merged axes for HMF and PS spanning two columns each
        ax_hmf = plt.subplot2grid((nrows, ncols), (0, 0), colspan=hmf_colspan)
        ax_ps = plt.subplot2grid((nrows, ncols), (1, 0), colspan=ps_colspan)

        # Prepare axes for histograms and density fields
        axs_hist = [plt.subplot2grid((nrows, ncols), (0, hmf_colspan + i)) for i in range(self.runs_num)]
        axs_dens = [plt.subplot2grid((nrows, ncols), (1, ps_colspan + i)) for i in range(self.runs_num)]

        colorbar_max = 6

        # Plot data for each run
        for i in range(self.runs_num):
            # Plot HMF
            ax_hmf.plot(self.bin_edges_runs[i], self.hist_runs[i],
                        marker='.', linestyle='-', color=self.plots_colors[i],
                        label=self.run_labels[i])

            # Plot 2D halo positions
            X, Y = np.meshgrid(self.xedges_runs[i], self.yedges_runs[i])
            mesh = axs_hist[i].pcolormesh(X, Y, self.halo_hist_runs[i].T, shading='auto', cmap='Reds')
            plt.colorbar(mesh, ax=axs_hist[i], label='Count')
            mesh.set_clim(0, colorbar_max)
            axs_hist[i].set_title(f"Halo Positions: {self.run_labels[i]}")

            # Plot density field
            self.runs[i].plot_field_slice(fig, axs_dens[i], field_type='rhog', intercept=0)
            axs_dens[i].set_title(f"Density: {self.run_labels[i]}")

            # Plot power spectrum
            ax_ps.plot(self.runs_psx[i], self.runs_psy[i],
                       marker='.', linestyle='-', color=self.plots_colors[i],
                       label=self.run_labels[i])

        # Set name for y-axis of HMF
        if hmf_type == 'dndlogm':
            hmf_y_label = "$\mathrm{d} n / \mathrm{d}  \log M$"
        elif hmf_type == 'dn':
            hmf_y_label = "Number of halos"
        else:
            hmf_y_label = "HMF (unknown type)"

        # Format HMF and PS axes
        ax_hmf.set(xlabel="$\log M$, (Mpc)", ylabel=hmf_y_label,
                   xscale='log', yscale='log', title='Halo Mass Function')
        ax_hmf.legend()

        ax_ps.set(xlabel='k', ylabel='PS',
                  xscale='log', yscale='log', title='Power Spectra Comparison')
        ax_ps.legend()

        # Adjust layout and save
        plt.tight_layout()
        out_file = os.path.join(self.out_dir, out_name)
        plt.savefig(out_file)
        plt.show()
        print(f"Saved figure to: {out_file}")
        return 0

def make_plots_for_several_run_groups(date, redshifts_lists, runnos_list, boxsize_list):
    for i, runno in enumerate(runnos_list):
        redshifts_list = redshifts_lists[i]
        boxsize = boxsize_list[i]
        out_name = '2025/Halo_Analysis_All_Plots_' + str(boxsize) + '_Mpc.png'

        run_paths, run_labels = make_paths_labels(date, redshifts_list, runno)
        hmfs_class = compare_HMFs(run_paths, run_labels, out_dir)
        hmfs_class.plot_runs(out_name=out_name)

make_plots_for_several_run_groups(date, redshifts_lists, runnos_list, boxsize_list)
