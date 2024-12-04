import numpy as np
import camb
import matplotlib
import matplotlib.pyplot as plt
from camb import model
from classy import Class
matplotlib.use('Agg') # Make plots interactive

# Only options are:
# 2 (Debug, TF_CDM vs k), 
# 7 (old one, whose purpose is not clear to me), 
# 13 (new one, accepted by MUSIC)
output_type = 13 
output_file = 'output.dat'
camb_file = 'CAMB.dat'
class_file = 'CLASS.dat'
figure_path = 'CLASS_vs_CAMB.png'
calc_nonG = True # Whether to claculate nongaussianities with CAMB
TF_src = 'CLASS' # Use CAMB or CLASS to generate transfer funtions

# Run parameters
class Run_params():
    """Class that contains its run parameters"""
    def __init__(self):
        # Cosmology parameters
        self.h        = 0.6735
        self.H0       = 100*self.h
        self.omc      = 0.2607     # Omega_cdm
        self.omb      = 0.04897    # Omega_baryon
        self.omk      = 0.0
        self.mnu      = 0.06
        self.tau      = 0.0544
        self.ns       = 0.9649
        self.As       = 2.1e-9
        self.sigma8   = 0.8111
        self.omch2    = self.omc * self.h**2 # Omega_cdm * h^2
        self.ombh2    = self.omb * self.h**2 # Omega_baryon * h^2
        # Output powerspectrum parameters
        self.nkpoints = 1000
        self.minkh    = 1e-4
        self.maxkh    = 5e3
        self.redshift = 0

#----------------------------------------------------------------------------
#-------------------------- END OF MODIFYABLE PART -------------------------- 
#----------------------------------------------------------------------------

class Transfer_data():
    """Class that contains transfer functions data"""
    def __init__(self):
        self.kh = np.array([])
        self.delta_cdm = np.array([])
        self.delta_b = np.array([])
        self.delta_g = np.array([])
        self.delta_nu = np.array([])
        self.delta_num = np.array([])
        self.delta_tot = np.array([])
        self.delta_nonu = np.array([])
        self.delta_totde = np.array([])
        self.phi = np.array([])
        self.v_cdm = np.array([])
        self.v_b = np.array([])
        self.v_b_cdm = np.array([])
        self.norm = 0.0

def create_TF_CAMB(params, calc_nonG):
    """
    Creates transfer functios using CAMB

    Args:
        params (Run_params): cosmology and power spectrum parameters
        calc_nonG (bool): a flag whether to calculate nongaussianities

    Returns:
        td (Transfer_data): TFs
    """
    # Set up the parameters
    pars = camb.CAMBparams()
    # Set up the transfer data class
    td = Transfer_data()
    # omch2 and ombh2 are already Omega_cdm * h^2 and Omega_b * h^2
    pars.set_cosmology(H0=params.H0, omch2=params.omch2, ombh2=params.ombh2, omk=params.omk, mnu=params.mnu, tau=params.tau)
    pars.InitPower.set_params(As=params.As, ns=params.ns)
    pars.set_matter_power(redshifts=[params.redshift], kmax=params.maxkh)
    pars.NonLinear = model.NonLinear_none # Used in PeakPatch
    #pars.NonLinear = model.NonLinear_both # Used in my MUSIC-generated ICs
    pars.PK_WantTransfer = 1  # Calculate the matter power transfer function
    pars.WantTransfer = 1
    pars.Transfer.kmax = params.maxkh * params.h
    
    # Calculate the results
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=params.minkh, maxkh=params.maxkh, npoints=params.nkpoints)
    s8 = np.array(results.get_sigma8())
    
    # Normalize matter power spectrum to match desired sigma_8
    print("sigma_8 pre normalization = ", s8)
    #norm = 1 # Normalization constant if you don't want normalization
    td.norm = (params.sigma8 / s8)**2  # Normalization constant
    
    if calc_nonG:
        k = kh * params.h               # Wavenumber k in Mpc^-1
        pk = td.norm * pk[0, :]   # Normalized P_m(z=0,k)

        # Primordial zeta power spectrum
        ko = 0.05
        pkzeta = 2 * np.pi**2 * params.As / k**3 * (k / ko)**(params.ns - 1)
        
        # Get transfer function
        Trans = np.sqrt(pk / pkzeta)
    
    # Extracting transfer functions
    transfer = results.get_matter_transfer_data()
    td.kh = transfer.transfer_data[0,:,0]  # Wavenumber k/h in Mpc^-1
    td.delta_cdm = transfer.transfer_data[1,:,0]  # CDM density contrast
    td.delta_b = transfer.transfer_data[2,:,0]    # Baryon density contrast
    td.delta_g = transfer.transfer_data[3,:,0]    # Photon density contrast
    td.delta_nu = transfer.transfer_data[4,:,0]   # Massless neutrinos
    td.delta_num = transfer.transfer_data[5,:,0]  # Massive neutrinos
    td.delta_tot = transfer.transfer_data[6,:,0]  # Total matter density contrast (including massive neutrinos)
    td.delta_nonu = transfer.transfer_data[7,:,0] # Total matter excluding neutrinos
    td.delta_totde = transfer.transfer_data[8,:,0] # Total including DE perturbations
    td.phi = transfer.transfer_data[6,:,0]        # Weyl potential
    td.v_cdm = transfer.transfer_data[10,:,0]     # Newtonian CDM velocity
    td.v_b = transfer.transfer_data[11,:,0]       # Newtonian baryon velocity
    td.v_b_cdm = transfer.transfer_data[12,:,0]   # Relative baryon-CDM velocity

    return td

def create_TF_CLASS(params):
    """
    Creates transfer functios using CLASS

    Args:
        params (Run_params): cosmology and power spectrum parameters

    Returns:
        td (Transfer_data): TFs
    """
    
    # Set up the transfer data class
    td = Transfer_data()

    # Set up CLASS
    LambdaCDM = Class()
    # pass input parameters
    h = params.h
    LambdaCDM.set({'omega_b':  params.omb*h**2, # Little omega, so omega = Omega * h^2
                   'omega_cdm':params.omc*h**2, # Little omega, so omega = Omega * h^2
                   'h':        h,
                   'A_s':      params.As,
                   'n_s':      params.ns,
                   'tau_reio': params.tau})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'k_min_tau0':params.minkh,
                   'z_pk': 0,
                   'P_k_max_h/Mpc':params.maxkh,
                   })
    LambdaCDM.compute()
    td.norm = (params.sigma8 / LambdaCDM.sigma8())**2  # Normalization constant
    kk = np.logspace(np.log10(params.minkh),np.log10(params.maxkh),params.nkpoints) # k in h/Mpc
    Pk = [] # P(k) in (Mpc/h)**3
    h = LambdaCDM.h() # get reduced Hubble for conversions to 1/Mpc
    for k in kk:
        Pk.append(LambdaCDM.pk(k*h,0.)*h**3) # function .pk(k,z)
    Tk = np.sqrt(Pk / kk**params.ns)
    td.kh = kk
    td.delta_cdm = Tk
    return td

def Renormalize_transfer(td):
    """
    Renormalizes transfer functions

    Args:
        td (Transfer_data): non-renormalized TFs

    Returns:
        td (Transfer_data): renormalized TFs
    """
    sqrt_norm = np.sqrt(td.norm)
    td.delta_cdm = td.delta_cdm * sqrt_norm 
    td.delta_b = td.delta_b * sqrt_norm 
    td.delta_g = td.delta_g * sqrt_norm 
    td.delta_nu = td.delta_nu * sqrt_norm 
    td.delta_num = td.delta_num * sqrt_norm 
    td.delta_tot = td.delta_tot * sqrt_norm 
    td.delta_nonu = td.delta_nonu * sqrt_norm 
    td.delta_totde = td.delta_totde * sqrt_norm 
    td.phi = td.phi * sqrt_norm 
    td.v_cdm = td.v_cdm * sqrt_norm 
    td.v_b = td.v_b * sqrt_norm 
    td.v_b_cdm = td.v_b_cdm * sqrt_norm 

    return td

def get_formatted_data(td, params, output_type):
    """
    formats the data in the format that is requested

    Args:
        td (Transfer_data): class that contains all the transfer functions
        params (Run_params): cosmology and power spectrum parameters
        output_type (int): a type of output requested. 13 for MUSIC, 2 for Debug

    Returns:
        header (str): header for output table
        data (np array): a table with all the transfer functions in output format
    """
    if output_type == 13: # MUSIC format
        header = 'k, delta_cdm, delta_b, delta_g, delta_nu, delta_num, delta_tot, delta_nonu, delta_totde, phi, v_cdm, v_b, v_b_cdm'
        data = np.vstack([td.kh * params.h,
                          td.delta_cdm,
                          td.delta_b,
                          td.delta_g,
                          td.delta_nu,
                          td.delta_num,
                          td.delta_tot,
                          td.delta_nonu,
                          td.delta_totde,
                          td.phi,
                          td.v_cdm,
                          td.v_b,
                          td.v_b_cdm
                        ]).T
        return header, data
    elif output_type == 2: # Debug format
        header = 'k, delta_cdm'
        data = np.vstack([td.kh * params.h,
                          td.delta_cdm,
                        ]).T
        return header, data

    print(f"Unsupported output type: {output_type}")
    exit(1)

def Save_output(header, data, output_file):
    """
    Save the output table

    Args:
        header (str): header for output table 
        data (np array): a table with all the transfer functions in output format
        output_file (str): name of the output table file

    Returns:
        int: 0 on success
    """
    # Avoid saving the output file if the output type is unsupported
    np.savetxt(output_file, data, header=header, fmt='%0.8e')
    print(f"Data saved to {output_file}")
    return 0

def create_and_save_TF(output_file, TF_src, output_type, calc_nonG=False):
    """
    This function is fully responsible for creating and saving transfer functions of a certain type, from start to finish

    Args:
        output_file (str): name of the output table file
        TF_src (str): what code will create the transfer functions (CAMB or CLASS)
        calc_nonG (bool): a flag whether to calculate nongaussianities
        output_type (int): a type of output requested. 13 for MUSIC, 2 for Debug

    Returns:
        data (np array): a table with all the transfer functions in output format
    """
    params = Run_params()
    if TF_src == 'CAMB':
        td = create_TF_CAMB(params, calc_nonG)
        td = Renormalize_transfer(td)
        print("Finished calculating CAMB TF")
    elif TF_src == 'CLASS':
        td = create_TF_CLASS(params)
        td = Renormalize_transfer(td)
        print("Finished calculating CLASS TF")
    else:
        print(f"Wrong TF_src variable value: {TF_src}. Allowed values are: CAMB, CLASS")
        exit(1)
    header, data = get_formatted_data(td, params, output_type)
    Save_output(header, data, output_file)
    return data

#data_camb  = create_and_save_TF(camb_file,  'CAMB',  2)
data_camb = np.loadtxt(camb_file, skiprows=1)
data_class = create_and_save_TF(class_file, 'CLASS', 2)

plt.plot(data_camb[:,0],  data_camb[:,1], label='CAMB, renormalized' )
plt.plot(data_class[:,0], data_class[:,1], label='CLASS')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig(figure_path)