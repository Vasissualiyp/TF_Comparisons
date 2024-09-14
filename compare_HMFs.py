usr_path = '/home/vasilii/research/software_src/'
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, usr_path+'peakpatch/python')
from peakpatchtools import PeakPatch



# ------------------ PARTS CHANGING BEGIN ----------------------
#run1_path = usr_path + 'pp_runs/interface_run1ic_run1/'
run1_path = usr_path + 'pp_runs/interface_hpkvd_run_control/'
run2_path = usr_path + 'pp_runs/interface_hpkvd_run1/'

run1_label = 'PeakPatch (Good)'
run2_label = 'PeakPatch (IPR)'
# ------------------ PARTS CHANGING END ------------------------






run1 = PeakPatch(run_dir=run1_path, params_file=run1_path+'param/parameters.ini')
run2 =  PeakPatch(run_dir=run2_path, params_file=run2_path+'param/parameters.ini')
# END IMPORT RUN

# Adding halos for both runs
run1.add_halos()
run2.add_halos()
print('Halos added for both runs')

# Calculating halo mass functions
hist_run1, bin_edges_run1 = run1.hmf(hmf_type='dn')
hist_run2, bin_edges_run2 = run2.hmf(hmf_type='dn')
print('Histogram parameters found for both runs')

# Calculating 2D histograms for halo properties
halo_hist_run1, xedges_run1, yedges_run1 = run1.halo_hist2d()
halo_hist_run2, xedges_run2, yedges_run2 = run2.halo_hist2d()
print('Halo histogram found for both runs')

# Adjust bin edges by wrapping around the periodic boundary conditions
box_size1 = run1.boxsize
box_size2 = run2.boxsize
half_boxsize1 = box_size1 // 2
half_boxsize2 = box_size2 // 2
#xedges_run1 = (xedges_run1 - half_boxsize1) % box_size1
#yedges_run1 = (yedges_run1 - half_boxsize1) % box_size1
#xedges_run2 = (xedges_run2 - half_boxsize2) % box_size2
#yedges_run2 = (yedges_run2 - half_boxsize2) % box_size2

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # 3 plots in one column

# Subplot 1: Halo Mass Function Comparison
axs[0].plot(bin_edges_run1[1:], hist_run1, marker='.', linestyle='-', color='red', label=run1_label)
axs[0].plot(bin_edges_run2[1:], hist_run2, marker='.', linestyle='-', color='blue', label=run2_label)
axs[0].set_xlabel('Mass Bin')
axs[0].set_ylabel('Number of Halos')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title('Halo Mass Function')
axs[0].legend()

# Subplot 2: 2D Histogram for MUSIC
X, Y = np.meshgrid(xedges_run1, yedges_run1)
hist2d_run1 = axs[1].pcolormesh(X, Y, halo_hist_run1.T, shading='auto', cmap='Reds')
fig.colorbar(hist2d_run1, ax=axs[1], label='Count')
axs[1].set_xlabel('X Property')
axs[1].set_ylabel('Y Property')
axs[1].set_title(f"2D Histogram of Halo Properties for {run1_label}")

# Subplot 3: 2D Histogram for PeakPatch
X, Y = np.meshgrid(xedges_run2, yedges_run2)
hist2d_run2 = axs[2].pcolormesh(X, Y, halo_hist_run2.T, shading='auto', cmap='Blues')
fig.colorbar(hist2d_run2, ax=axs[2], label='Count')
axs[2].set_xlabel('X Property')
axs[2].set_ylabel('Y Property')
axs[2].set_title(f"2D Histogram of Halo Properties for {run2_label}")

# Save the entire figure
out_dir = os.getcwd()
out_file = os.path.join(out_dir, 'Halo_Analysis_All_Plots.png')
plt.savefig(out_file)
print(f"Saved all plots in {out_file}")

plt.show()
