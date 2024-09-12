usr_path = '/home/vasilii/research/software_src/'
import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, usr_path+'peakpatch/python')
from peakpatchtools import PeakPatch
run_mus_path = usr_path + 'pp_runs/interface_music_run1/'
run_pp_path = usr_path + 'pp_runs/interface_hpkvd_run1/'

run_mus = PeakPatch(run_dir=run_mus_path, params_file=run_mus_path+'param/parameters.ini')
run_pp =  PeakPatch(run_dir=run_pp_path, params_file=run_pp_path+'param/parameters.ini')
# END IMPORT RUN

# Get the histograms for MUSIC and hpkvd runs
run_mus.add_halos()
run_pp.add_halos()
print('Halos added for both runs')
hist_mus, bin_edges_mus = run_mus.hmf()
hist_pp, bin_edges_pp = run_pp.hmf()
print('Histogram parameters found for both runs')

# Create a histogram plot
plt.figure(figsize=(10, 6))  # Adjust the size of the figure
#plt.bar(bin_edges_mus[:-1], hist_mus width=np.diff(bin_edges_mus), edgecolor='black', align='edge')
plt.plot(bin_edges_mus[1:], hist_mus, marker='.', linestyle='-', color='red',label='MUSIC')
plt.plot(bin_edges_pp[1:], hist_pp, marker='.', linestyle='-', color='blue',label='PeakPatch')

# Set labels and title
plt.xlabel('Mass Bin')
plt.ylabel('Number of Halos')
plt.xscale('log')
plt.yscale('log')
plt.title('Halo Mass Function')
plt.legend()

# Save the plot to a file
out_dir=usr_path + 'pp_runs/python/'
out_file = out_dir + 'HMF_ntile2_nmesh256.png'
plt.savefig(out_file)
print(f"Saved output in ", out_file)

