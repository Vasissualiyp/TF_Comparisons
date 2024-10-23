usr_path = '/home/vasilii/research/software_src/'
usr_path = '../../peakpatch/python/'
import os, sys, numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
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


#run2_label = 'PeakPatch (IPR)'

# ------------------ PARTS CHANGING BEGIN ----------------------
#run1_path = usr_path + 'pp_runs/interface_run1ic_run1/'
run1_path = total_usr_path + 'pp_runs/hpkvd-interface-run2/'
#run1_path = usr_path + 'data/2024-09/cambridge_run_2048/'
#run2_path = usr_path + 'data/2024-09/cambridge_run_2048/'
#run2_path = total_usr_path + 'pp_runs/music-interface-run-bbks/'
run2_path = total_usr_path + 'pp_runs/music-interface-run/'
run2_path = total_usr_path + 'pp_runs/music-interface-run2/'

#run1_label = 'PeakPatch (Good)'
#run1_label = "z=11(?) 2048^3 cells 75 Mpc run (Rsmooth_max=1.577)"
run1_label = "hpkvd run"
#run2_label = 'PeakPatch (IPR)'
#run2_label = 'z=11(?) 4096^3 cells 6.4 Mpc run (Rsmooth_max=0.0668)'
#run2_label = "music run (BBKS)"
run2_label = "music run"
# ------------------ PARTS CHANGING END ------------------------


if machine == 'cita':
    field_file_run1 = run1_path + "/fields/Fvec_640Mpc_Cambridge"
    field_file_run2 = run2_path + "/fields/Fvec_640Mpc_MUSIC"
elif machine == 'vas':
    field_file_run1 = sims_dir + "pp_runs/hpkvd-interface-run/fields/Fvec_640Mpc_Cambridge"
    field_file_run2 = sims_dir + "pp_runs/music-interface-run2/fields/Fvec_640Mpc_MUSIC"
else:
    print(f"Unknown machine: {machine}. Allowed values: cita, vas")




run1 = PeakPatch(run_dir=run1_path, params_file=run1_path+'param/parameters.ini')
run2 = PeakPatch(run_dir=run2_path, params_file=run2_path+'param/parameters.ini')
# END IMPORT RUN

# Adding halos for both runs
run1.add_halos()
run2.add_halos()
print('Halos added for both runs')

# Calculating halo mass functions
hist_run1, bin_edges_run1 = run1.hmf(hmf_type='dn')
hist_run2, bin_edges_run2 = run2.hmf(hmf_type='dn')
print('Histogram parameters found for both runs')

run1.add_field(field_type='rhog')
run2.add_field(field_type='rhog')
print('Density fields added for both runs')

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
fig, axs = plt.subplots(2, 3, figsize=(20, 12))  # 3 plots in one column

# Subplot 1: Halo Mass Function Comparison
axs[0,0].plot(bin_edges_run1[1:], hist_run1, marker='.', linestyle='-', color='red', label=run1_label)
axs[0,0].plot(bin_edges_run2[1:], hist_run2, marker='.', linestyle='-', color='blue', label=run2_label)
axs[0,0].set_xlabel('Mass Bin')
axs[0,0].set_ylabel('Number of Halos')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].set_title('Halo Mass Function')
axs[0,0].legend()

# Subplot 2: 2D Histogram for MUSIC
X, Y = np.meshgrid(xedges_run1, yedges_run1)
hist2d_run1 = axs[0,1].pcolormesh(X, Y, halo_hist_run1.T, shading='auto', cmap='Reds')
fig.colorbar(hist2d_run1, ax=axs[0,1], label='Count')
axs[0,1].set_xlabel('X')
axs[0,1].set_ylabel('Y')
axs[0,1].set_title(f"2D Histogram of halo positions for {run1_label}")

# Subplot 3: 2D Histogram for PeakPatch
X, Y = np.meshgrid(xedges_run2, yedges_run2)
hist2d_run2 = axs[0,2].pcolormesh(X, Y, halo_hist_run2.T, shading='auto', cmap='Blues')
fig.colorbar(hist2d_run2, ax=axs[0,2], label='Count')
axs[0,2].set_xlabel('X')
axs[0,2].set_ylabel('Y')
axs[0,2].set_title(f"2D Histogram of halo positions for {run2_label}")

#field_file_run1="/home/vasilii/research/sims/PeakPatch/pp_runs/hpkvd-interface-run/fields/Fvec_640Mpc_Cambridge"
#field_file_run2="/home/vasilii/research/sims/PeakPatch/pp_runs/music-interface-run/fields/Fvec_640Mpc_MUSIC"

# Overwrite MUSIC, don't overwrite HPKVD
run1.get_power_spectrum(field_file=field_file_run1, field_type='rhog', overwrite = False)
run2.get_power_spectrum(field_file=field_file_run2, field_type='rhog', overwrite = True)
run1.plot_field_slice(fig, axs[1,1], field_type='rhog', intercept=0)
run2.plot_field_slice(fig, axs[1,2], field_type='rhog', intercept=0)

run1_psx = run1.k_from_rhog
run1_psy = run1.k_from_rhog**3 / (2 * np.pi**2) * run1.p_from_rhog
run2_psx = run2.k_from_rhog
run2_psy = run2.k_from_rhog**3 / (2 * np.pi**2) * run2.p_from_rhog

# Subplot 4: Halo Mass Function Comparison
axs[1,0].plot(run1_psx, run1_psy, marker='.', linestyle='-', color='red', label=run1_label)
axs[1,0].plot(run2_psx, run2_psy, marker='.', linestyle='-', color='blue', label=run2_label)
axs[1,0].set_xlabel('k')
axs[1,0].set_ylabel('PS')
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].set_title('Power Spectra comparison')
axs[1,0].legend()

axs[1,1].set_title(f"Density field, {run1_label}")
axs[1,2].set_title(f"Density field, {run2_label}")

## Subplot 5: 2D Histogram for MUSIC
#X, Y = np.meshgrid(xedges_run1, yedges_run1)
#dhist2d_run1 = axs[0,1].pcolormesh(X, Y, del_lin_hist_run1.T, shading='auto', cmap='Reds')
#fig.colorbar(dhist2d_run1, ax=axs[0,1], label='Count')
#axs[1,1].set_xlabel('X')
#axs[1,1].set_ylabel('Y')
#axs[1,1].set_title(f"2D Histogram of Density for {run1_label}")
#
## Subplot 6: 2D Histogram for PeakPatch
#X, Y = np.meshgrid(xedges_run2, yedges_run2)
#dhist2d_run2 = axs[0,2].pcolormesh(X, Y, del_lin_hist_run2.T, shading='auto', cmap='Blues')
#fig.colorbar(dhist2d_run2, ax=axs[0,2], label='Count')
#axs[1,2].set_xlabel('X')
#axs[1,2].set_ylabel('Y')
#axs[1,2].set_title(f"2D Histogram of Density for {run2_label}")

out_file = os.path.join(out_dir, 'Halo_Analysis_All_Plots.png')
plt.show()
plt.savefig(out_file)
