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
run3_path = total_usr_path + 'pp_runs/nong_test_z11/'

#run1_label = 'PeakPatch (Good)'
#run1_label = "z=11(?) 2048^3 cells 75 Mpc run (Rsmooth_max=1.577)"
run1_label = "hpkvd run z=0"
#run2_label = 'PeakPatch (IPR)'
#run2_label = 'z=11(?) 4096^3 cells 6.4 Mpc run (Rsmooth_max=0.0668)'
#run2_label = "music run (BBKS)"
run2_label = "hpkvd run z=5"
run3_label = "hpkvd run z=11"
# ------------------ PARTS CHANGING END ------------------------

run_paths =  [ run1_path,  run2_path,  run3_path  ]
run_labels = [ run1_label, run2_label, run3_label ]

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

    def plot_runs():
        for i in range(runs_num):
            self.runs.append( PeakPatch(run_dir=run_paths[i], params_file=run_paths[i]+'param/parameters.ini') )
        
            # Adding halos for run {i}
            self.runs[i].add_halos()
            print(f"Run {i}: Halos added")
        
            # Calculating halo mass functions
            hist_run, bin_edges_run = self.runs[i].hmf(hmf_type='dn')
            self.hist_runs.append(hist_run)
            self.bin_edges_runs.append(bin_edges_run)
            #hist_run2, bin_edges_run2 = self.runs[1].hmf(hmf_type='dn')
            print(f"Run {i}: Histogram parameters found")
        
            self.runs[i].add_field(field_type='rhog')
            #self.runs[1].add_field(field_type='rhog')
            print(f"Run {i}: Density fields added")
        
            # Calculating 2D histograms for halo properties
            halo_hist_run, xedges_run, yedges_run = self.runs[i].halo_hist2d()
            self.halo_hist_runs.append(halo_hist_run)
            self.xedges_runs.append(xedges_run)
            self.yedges_runs.append(yedges_run)
            #halo_hist_run2, xedges_run2, yedges_run2 = self.runs[1].halo_hist2d()
            print(f"Run {i}: Halo histogram found")
        
        # Create a figure with subplots
        nrows = 2
        ncols = runs_num + 1
        hsize, vsize = get_figsize(nrows, ncols, 4, 2)
    
        colorbar_max = 12 # sets the maximum value for the colorbar for halo plotting
    
        fig, axs = plt.subplots(nrows, ncols, figsize=(hsize, vsize))
    
        print("Starting plotting...")
        
        # Subplot 1: Halo Mass Function Comparison
        for i in range(runs_num):
            plotid = 1 + i
            axs[0,0].plot(self.bin_edges_runs[i][1:], self.hist_runs[i], marker='.', linestyle='-', 
                          color=self.plots_colors[i],  label=run_labels[i])
    
            # Subplots 2-(N+1): 2D Histogram for MUSIC
            X, Y = np.meshgrid(self.xedges_runs[i], self.yedges_runs[i])
            self.hist2d_runs.append(axs[0,plotid].pcolormesh(X, Y, self.halo_hist_runs[i].T, shading='auto', cmap='Reds'))
            fig.colorbar(self.hist2d_runs[i], ax=axs[0,plotid], label='Count')
            self.hist2d_runs[i].set_clim(vmin=0, vmax=colorbar_max)
            axs[0,plotid].set_xlabel('X')
            axs[0,plotid].set_ylabel('Y')
            axs[0,plotid].set_title(f"2D Histogram of halo positions for {run_labels[i]}")
            print(f"Run {i}: Plotted Halos Histogram")
    
            # Subplots (N+3)-(2N): Overdensity fields
            axs[1,plotid].set_title(f"Density field, {run_labels[i]}")
    
            # Set which label to overwrite
            overwrite = False
            if run_labels[i] == "hpkvd run z=11":
                overwrite = True
    
            self.runs[i].get_power_spectrum(field_type='rhog', overwrite = overwrite)
            self.runs[i].plot_field_slice(fig, axs[1,plotid], field_type='rhog', intercept=0)
    
            # Subplot (N+2): Halo Mass Function Comparison
            self.runs_psx.append( self.runs[i].k_from_rhog )
            self.runs_psy.append( self.runs[i].k_from_rhog**3 / (2 * np.pi**2) * self.runs[i].p_from_rhog )
            axs[1,0].plot(self.runs_psx[i], self.runs_psy[i], marker='.', 
                          linestyle='-', color=self.plots_colors[i], 
                          label=run_labels[i])
            print(f"Run {i}: Plotted Overdensity")
    
        # Format Subplot 1 (HMFs)
        axs[0,0].set_xlabel('Mass Bin')
        axs[0,0].set_ylabel('Number of Halos')
        axs[0,0].set_xscale('log')
        axs[0,0].set_yscale('log')
        axs[0,0].set_title('Halo Mass Function')
        axs[0,0].legend()
    
        # Format Subplot (N+1) (Power Spectra)
        axs[1,0].set_xlabel('k')
        axs[1,0].set_ylabel('PS')
        axs[1,0].set_xscale('log')
        axs[1,0].set_yscale('log')
        axs[1,0].set_title('Power Spectra comparison')
        axs[1,0].legend()
        
        # Save figure output
        out_file = os.path.join(out_dir, 'Halo_Analysis_All_Plots.png')
        plt.show()
        plt.savefig(out_file)
        print(f"Saved figure to: {out_file}")
        return 0

plot_runs(run_paths, run_labels, out_dir)
