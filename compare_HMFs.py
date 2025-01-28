usr_path = '/home/vasilii/research/software_src/'
usr_path = '../../peakpatch/python/'
import os, sys, math, numpy as np
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


#run2_label = 'PeakPatch (IPR)'

# ------------------ PARTS CHANGING BEGIN ----------------------
#run1_path = usr_path + 'pp_runs/interface_run1ic_run1/'
run1_path = total_usr_path + 'pp_runs/hpkvd-interface-run3/'
#run1_path = usr_path + 'data/2024-09/cambridge_run_2048/'
#run2_path = usr_path + 'data/2024-09/cambridge_run_2048/'
#run2_path = total_usr_path + 'pp_runs/music-interface-run-bbks/'
#run2_path = total_usr_path + 'pp_runs/music-interface-run/'
#run2_path = total_usr_path + 'pp_runs/music-interface-run6/'
run2_path = total_usr_path + 'pp_runs/nong_test_z5/'
run3_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z8/'
run4_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z11/'
run5_path = total_usr_path + 'pp_runs/2025-01-27-niagara/z13/'

#run1_label = 'PeakPatch (Good)'
#run1_label = "z=11(?) 2048^3 cells 75 Mpc run (Rsmooth_max=1.577)"
run1_label = "hpkvd run z=0"
#run2_label = 'PeakPatch (IPR)'
#run2_label = 'z=11(?) 4096^3 cells 6.4 Mpc run (Rsmooth_max=0.0668)'
#run2_label = "music run (BBKS)"
run2_label = "hpkvd run z=5"
run3_label = "hpkvd run z=8"
run4_label = "hpkvd run z=11"
run5_label = "hpkvd run z=13"
# ------------------ PARTS CHANGING END ------------------------

run_paths =  [ run1_path,  run2_path,  run3_path,  run4_path,  run5_path  ]
run_labels = [ run1_label, run2_label, run3_label, run4_label, run5_label ]

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

    def plot_runs(self, hmf_colspan=2):
        # Initialize lists to store run data
        self.hist_runs = []
        self.bin_edges_runs = []
        self.halo_hist_runs = []
        self.xedges_runs = []
        self.yedges_runs = []
        self.runs_psx = []
        self.runs_psy = []

        ps_colspan = hmf_colspan

        # Process each run first (compute HMFs, fields, etc.)
        for i in range(self.runs_num):
            # Load run data
            run = PeakPatch(run_dir=self.run_paths[i], params_file=self.run_paths[i] + 'param/parameters.ini')
            run.add_halos()
            print(f"Run {i}: Halos added")

            # Compute Halo Mass Function
            hist_run, bin_edges_run = run.hmf(hmf_type='dn')
            self.hist_runs.append(hist_run)
            self.bin_edges_runs.append(bin_edges_run)
            print(f"Run {i}: HMF computed")

            # Compute density field and power spectrum
            run.add_field(field_type='rhog')
            run.get_power_spectrum(field_type='rhog', overwrite=(self.run_labels[i] == "hpkvd run z=11"))
            self.runs_psx.append(run.k_from_rhog)
            self.runs_psy.append(run.k_from_rhog**3 / (2 * np.pi**2) * run.p_from_rhog)

            # Compute 2D halo histogram
            halo_hist_run, xedges_run, yedges_run = run.halo_hist2d()
            self.halo_hist_runs.append(halo_hist_run)
            self.xedges_runs.append(xedges_run)
            self.yedges_runs.append(yedges_run)
            print(f"Run {i}: 2D halo histogram computed")

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

        colorbar_max = 12

        # Plot data for each run
        for i in range(self.runs_num):
            # Plot HMF
            ax_hmf.plot(self.bin_edges_runs[i][1:], self.hist_runs[i],
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

        # Format HMF and PS axes
        ax_hmf.set(xlabel='Mass Bin', ylabel='Number of Halos',
                   xscale='log', yscale='log', title='Halo Mass Function')
        ax_hmf.legend()

        ax_ps.set(xlabel='k', ylabel='PS',
                  xscale='log', yscale='log', title='Power Spectra Comparison')
        ax_ps.legend()

        # Adjust layout and save
        plt.tight_layout()
        out_file = os.path.join(self.out_dir, 'Halo_Analysis_All_Plots.png')
        plt.savefig(out_file)
        plt.show()
        print(f"Saved figure to: {out_file}")
        return 0

hmfs_class = compare_HMFs(run_paths, run_labels, out_dir)
hmfs_class.plot_runs()
