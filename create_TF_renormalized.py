import numpy as np
import camb
from camb import model

# Only options are:
# 7 (old one, whose purpose is not clear to me), 
# 13 (new one, accepted by MUSIC)
output_type = 13 
output_file = 'output.dat'

# Run parameters
h        = 0.6735
H0       = 100*h
omch2    = 0.2607   # Omega_cdm * h^2
ombh2    = 0.04897  # Omega_baryon * h^2
omk      = 0.0
mnu      = 0.0
tau      = 0.0544
ns       = 0.9649
As       = 2.1e-9
sigma8   = 0.8111
m_nu     = 0.0

redshift = 0

# Output powerspectrum parameters
nkpoints = 1000
minkh    = 5e-6
maxkh    = 5e3

# Scaling factor for units
camb_factor = (2 * np.pi * h)**3

# Set up the parameters
pars = camb.CAMBparams()
# omch2 and ombh2 are already Omega_cdm * h^2 and Omega_b * h^2
pars.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2, omk=omk, mnu=mnu, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_matter_power(redshifts=[redshift], kmax=maxkh)
pars.NonLinear = model.NonLinear_both
pars.PK_WantTransfer = True  # Calculate the matter power transfer function
pars.WantTransfer = True
pars.Transfer.kmax = maxkh

# Calculate the results
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=nkpoints)
s8 = np.array(results.get_sigma8())

# Normalize matter power spectrum to match desired sigma_8
print("sigma_8 pre normalization = ", s8)
norm = (sigma8 / s8)**2  # Normalization constant
k = kh * h               # Wavenumber k in Mpc^-1
pk = norm * pk[0, :] / camb_factor  # Normalized P_m(z=0,k)

# Primordial zeta power spectrum
ko = 0.05
pkzeta = 2 * np.pi**2 * As / k**3 * (k / ko)**(ns - 1)

# Get transfer function
Trans = np.sqrt(pk / pkzeta)

# Extracting transfer functions
transfer = results.get_matter_transfer_data()
kh = transfer.transfer_data[0,:,0]  # Wavenumber k/h in Mpc^-1
delta_cdm = transfer.transfer_data[1,:,0]  # CDM density contrast
delta_b = transfer.transfer_data[2,:,0]    # Baryon density contrast
delta_g = transfer.transfer_data[3,:,0]    # Photon density contrast
delta_nu = transfer.transfer_data[4,:,0]   # Massless neutrinos
delta_num = transfer.transfer_data[5,:,0]  # Massive neutrinos
delta_tot = transfer.transfer_data[5,:,0]  # Total matter density contrast (including massive neutrinos)
delta_nonu = transfer.transfer_data[7,:,0] # Total matter excluding neutrinos
delta_totde = transfer.transfer_data[8,:,0] # Total including DE perturbations
phi = transfer.transfer_data[6,:,0]        # Weyl potential
v_cdm = transfer.transfer_data[10,:,0]     # Newtonian CDM velocity
v_b = transfer.transfer_data[11,:,0]       # Newtonian baryon velocity
v_b_cdm = transfer.transfer_data[12,:,0]   # Relative baryon-CDM velocity

# Adjust transfer functions with normalization and units
sqrt_norm = np.sqrt(norm)
delta_cdm = delta_cdm * sqrt_norm / camb_factor
delta_b = delta_b * sqrt_norm / camb_factor
delta_g = delta_g * sqrt_norm / camb_factor
delta_nu = delta_nu * sqrt_norm / camb_factor
delta_num = delta_num * sqrt_norm / camb_factor
delta_tot = delta_tot * sqrt_norm / camb_factor
delta_nonu = delta_nonu * sqrt_norm / camb_factor
delta_totde = delta_totde * sqrt_norm / camb_factor
phi = phi * sqrt_norm / camb_factor
v_cdm = v_cdm * sqrt_norm / camb_factor
v_b = v_b * sqrt_norm / camb_factor
v_b_cdm = v_b_cdm * sqrt_norm / camb_factor

# Save to .dat file - MUSIC format
if output_type == 13:
    header = 'k, delta_cdm, delta_b, delta_g, delta_nu, delta_num, delta_tot, delta_nonu, delta_totde, phi, v_cdm, v_b, v_b_cdm'
    data = np.vstack([kh * h,
                      delta_cdm,
                      delta_b,
                      delta_g,
                      delta_nu,
                      delta_num,
                      delta_tot,
                      delta_nonu,
                      delta_totde,
                      phi,
                      v_cdm,
                      v_b,
                      v_b_cdm
                    ]).T
else:
    print(f"Unsupported output type: {output_type}")

np.savetxt(output_file, data, header=header, fmt='%0.8e')
print(f"Data saved to {output_file}")
