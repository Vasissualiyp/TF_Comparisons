import numpy as np
# -- IMPORT CAMB PROPERLY HERE --
import camb
from camb import model
import matplotlib.pyplot as plt
# -------------------------------

# Only options are 7 (old one, whose purpose is not clear to me), 
# and 13 (new one, accepted by MUSIC)
output_type = 13 
output_file = 'non_renorm_out.dat'

# Run parameters
h = 0.6735
H0    = 100*h
omch2 = 0.2607  #* h**2 # Omega_cdm * h^2
ombh2 = 0.04897 #* h**2 # Omega_baryon * h^2
omk   = 0.0
mnu   = 0.0
tau   = 0.0544
redshift = 0

# A factor I played with for scaling
camb_factor = 1 # (2 * np.pi * h)**3

# Set up the parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, omk=omk, mnu=mnu, tau=tau)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
pars.set_matter_power(redshifts=[redshift], kmax=5000.0)
pars.NonLinear = model.NonLinear_both

# Calculate the results
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=5000, npoints=1000)

# Extracting velocity and potential data
transfer = results.get_matter_transfer_data()
k = transfer.transfer_data[0,:,0]  # Wavenumber k/h in Mpc-1
delta_cdm = transfer.transfer_data[1,:,0]  # CDM density contrast
delta_b = transfer.transfer_data[2,:,0]  # Baryon density contrast
delta_g = transfer.transfer_data[3,:,0]  # Photon density contrast
delta_nu = transfer.transfer_data[4,:,0]  # Massless neutrinos
delta_num = transfer.transfer_data[5,:,0]  # Massive neutrinos
delta_tot = delta_cdm + delta_b + delta_nu  # Total matter density contrast
delta_nonu = transfer.transfer_data[7,:,0] # Total matter excluding neutrinos
delta_totde = transfer.transfer_data[8,:,0] # Total includeing DE perturbations
phi = transfer.transfer_data[6,:,0]  # Weyl potential
v_cdm = transfer.transfer_data[10,:,0] # Newtonian CDM velocity
v_b = transfer.transfer_data[11,:,0] # Newtonian baryon velocity
v_b_cdm = transfer.transfer_data[12,:,0] # relative baryon-cdm velocity



# Save to .dat file - PeakPatch
if output_type == 7:
    header = 'k/h Delta_CDM/k2 Delta_b/k2 Delta_g/k2 Delta_nu/k2 Delta_tot/k2 Phi'
    data = np.vstack([k,
                      delta_cdm/k**2,
                      delta_b/k**2,
                      delta_g/k**2,
                      delta_nu/k**2,
                      delta_tot/k**2,
                      phi
                    ]).T
# Save to .dat file - MUSIC
elif output_type == 13:
    header = 'k, delta_cdm, delta_b, delta_g, delta_nu, delta_num, delta_tot, delta_nonu, delta_totde, phi, v_cdm, v_b, v_b_cdm'
    data = np.vstack([k * h,
                      delta_cdm/camb_factor,
                      delta_b/camb_factor,
                      delta_g/camb_factor,
                      delta_nu/camb_factor,
                      delta_num/camb_factor,
                      delta_tot/camb_factor,
                      delta_nonu/camb_factor,
                      delta_totde/camb_factor,
                      phi/camb_factor,
                      v_cdm/camb_factor,
                      v_b/camb_factor,
                      v_b_cdm/camb_factor
                    ]).T
else:
    print(f"Unsupported output type: {output_type}")


np.savetxt(output_file, data, header=header, fmt='%0.8e')
print(f"Data saved to {output_file}")
