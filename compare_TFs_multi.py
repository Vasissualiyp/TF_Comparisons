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
    sims_dir = usr_path_vas
    out_dir = "."
else:
    print(f"Unknown machine: {machine}. Allowed values: cita, vas")

# ------------------ PARTS CHANGING BEGIN ----------------------

run_paths = [
    total_usr_path + 'pp_runs/hpkvd-interface-run2/',
    total_usr_path + 'pp_runs/music-interface-run2/',
    total_usr_path + 'pp_runs/z0/',
    #total_usr_path + 'pp_runs/post_poisson_1/',
    #total_usr_path + 'pp_runs/post_poisson_2/',
    #total_usr_path + 'pp_runs/pre_poisson_1/',
    #total_usr_path + 'pp_runs/pre_poisson_2/',
    #total_usr_path + 'pp_runs/aperiodicTF/'
    #total_usr_path + 'pp_runs/TFmin_11/',
    #total_usr_path + 'pp_runs/TFmin_12/',
]
run_labels = [
    "hpkvd run",
    "music run, $\\mathcal{P} \\times 2$",
    "z=99, $\\mathcal{P} \\times (1+z)^2$ "
    #"post_poisson_1",
    #"post_poisson_2",
    #"pre_poisson_1",
    #"pre_poisson_2",
    #"aperiodicTF"
    #"TFmin_11",
    #"TFmin_12",
]

field_files = []
if machine == 'cita':
    field_files = [
        os.path.join(run_paths[0], "fields/Fvec_640Mpc_Cambridge"),
        os.path.join(run_paths[1], "fields/Fvec_640Mpc_MUSIC"),
        os.path.join(run_paths[2], "fields/Fvec_640Mpc_MUSIC"),
        #os.path.join(run_paths[3], "fields/Fvec_640Mpc_MUSIC"),
        #os.path.join(run_paths[4], "fields/Fvec_640Mpc_MUSIC"),
        #os.path.join(run_paths[5], "fields/Fvec_640Mpc_MUSIC"),
        #os.path.join(run_paths[6], "fields/Fvec_640Mpc_MUSIC")
        #os.path.join(run_paths[7], "fields/Fvec_640Mpc_MUSIC"),
        #os.path.join(run_paths[8], "fields/Fvec_640Mpc_MUSIC")
    ]
elif machine == 'vas':
    field_files = [
        os.path.join(sims_dir, "pp_runs/hpkvd-interface-run/fields/Fvec_640Mpc_Cambridge"),
        os.path.join(sims_dir, "pp_runs/music-interface-run2/fields/Fvec_640Mpc_MUSIC")
    ]
else:
    print(f"Unknown machine: {machine}. Allowed values: cita, vas")

adjustment_factors = [ 1, 2, 1e4 ]

# ------------------ PARTS CHANGING END ------------------------


# Flags to enable/disable plotting of HMF and TF
plot_HMF = False
plot_TF = True

# Create a list to store the PeakPatch runs
runs = []

for run_path, run_label in zip(run_paths, run_labels):
    run = PeakPatch(run_dir=run_path, params_file=run_path+'param/parameters.ini')
    run.label = run_label  # Assign label to the run object
    runs.append(run)

# If plotting HMF, add halos and compute HMFs
if plot_HMF:
    for run in runs:
        run.add_halos()
    print('Halos added for all runs')

    # Calculating halo mass functions
    hmf_hist_list = []
    hmf_bin_edges_list = []
    for run in runs:
        hist, bin_edges = run.hmf(hmf_type='dn')
        hmf_hist_list.append(hist)
        hmf_bin_edges_list.append(bin_edges)
    print('Histogram parameters found for all runs')

# If plotting TF, add field and compute power spectra
if plot_TF:
    for run, field_file in zip(runs, field_files):
        run.add_field(field_type='rhog')
    print('Density fields added for all runs')

    # Compute power spectra
    for run, field_file in zip(runs, field_files):
        run.get_power_spectrum(field_file=field_file, field_type='rhog', overwrite=False)
    print('Power spectra calculated for all runs')

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 plots in one row

# Plot HMF comparison
if plot_HMF:
    for run, hist, bin_edges in zip(runs, hmf_hist_list, hmf_bin_edges_list):
        axs[0].plot(bin_edges[1:], hist, marker='.', linestyle='-', label=run.label)
    axs[0].set_xlabel('Mass Bin')
    axs[0].set_ylabel('Number of Halos')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_title('Halo Mass Function')
    axs[0].legend()

# Plot TF comparison
if plot_TF:
    for i, run in enumerate(runs):
        adj_fac = adjustment_factors[i]
        psx = run.k_from_rhog
        psy = run.k_from_rhog**3 / (2 * np.pi**2) * run.p_from_rhog
        psy = psy * adj_fac
        axs[1].plot(psx, psy, marker='.', linestyle='-', label=run.label)
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('$\\mathcal{P}$')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_title('Power Spectra comparison')
    axs[1].legend()

out_file = os.path.join(out_dir, 'Halo_Analysis_HMF_TF_multi.png')
plt.show()
plt.savefig(out_file)

