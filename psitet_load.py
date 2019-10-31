## @package psitet_load
## Contains routines for loading in different file types
## from experimental, nimrod, and psi-tet data sources
from os import path
import scipy.io as sio
from psitet import psiObject
from dataclasses import asdict
from plot_attributes import *

## Loads a psi-tet mat file using the old format from Tom's scan paper
# @param fname Name of the base file
# @param runfolder Name of the folder where the file resides
# @param f_1 f_1 in kHz
# @param is_HITSI3 Flag to indicate if this is a HITSI3 run
# @returns psitet List of psi-tet dictionaries
def load_psitet_old(fname, runfolder, f_1, is_HITSI3, limits):
    num_sp_probes = 96
    num_imp_probes = 100
    num_inter_probes = 100
    sp = np.loadtxt(runfolder + 'sp_locations.txt', dtype='str')
    imp = np.loadtxt(runfolder + 'imp_locations.txt', dtype='float')
    sp_names = sp[:, 0]
    sp_R = sp[:, 1].astype(float)
    sp_Z = sp[:, 2].astype(float)
    sp_phi = sp[:, 3].astype(float)
    sp_theta = np.arctan2(sp_Z, sp_R - 33)
    B_theta = sp[:, 4].astype(float)
    # Load psi-tet 1T files
    mat_files = ['xmhd', 'hitn_driver', 'heat_flux', 'hitn_sp_probes', 'hitn_imp_probes',
                 'hitn_inter_probes', 'hitn_idsn_probes', 'hitn_idsT_probes', 'hitn_idsV_probes']
    p = psiObject()
    # This list is alphabetically sorted
    psilist = [a for a in dir(p) if not a.startswith('__')]
    p = asdict(p)
    for j in range(len(mat_files)):
        filename = runfolder + fname + str(f_1) + \
            '_' + mat_files[j] + '.mat'
        if path.exists(filename):
            A = sio.loadmat(filename)
            if mat_files[j] == 'xmhd':
                A['ti'] = A.pop('temp')
            if mat_files[j] == 'hitn_driver':
                A['driver_time'] = A.pop('time')
            elif mat_files[j] == 'heat_flux':
                A['heat_flux_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_sp_probes':
                A['sp_r'] = sp_R
                A['sp_z'] = sp_Z
                A['sp_phi'] = sp_phi
                A['sp_names'] = sp_names
                Bx = []
                By = []
                Bz = []
                B_pol = []
                B_tor = []
                for q in range(1, num_sp_probes + 1):
                    zero_str = '0' * (6 - len(str(q))) + str(q)
                    spx_str = 'd_' + zero_str + '_01'
                    spy_str = 'd_' + zero_str + '_02'
                    spz_str = 'd_' + zero_str + '_03'
                    Bx.append(A[spx_str])
                    By.append(A[spy_str])
                    Bz.append(A[spz_str])
                    tpos = sp_phi[q - 1] * pi / 180.0
                    pangle = B_theta[q - 1] * pi / 180.0
                    that = [-np.sin(tpos), np.cos(tpos), 0]
                    rhat = [np.cos(tpos), np.sin(tpos), 0]
                    B_pol.append(np.sin(pangle) * \
                        (rhat[0] * Bx[q - 1] + rhat[1] * By[q - 1]) + \
                        np.cos(pangle) * Bz[q - 1])
                    B_tor.append(that[0] * Bx[q - 1] + that[1] * By[q - 1])
                A['sp_Bx'] = Bx
                A['sp_By'] = By
                A['sp_Bz'] = Bz
                A['sp_Bpol'] = B_pol
                A['sp_Btor'] = B_tor
                A['sp_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_imp_probes':
                Bx = []
                By = []
                Bz = []
                B_pol = []
                B_tor = []
                B_rad = []
                R = []
                Z = []
                Phi = []
                for q in range(1, num_imp_probes + 1):
                    zero_str = '0' * (6 - len(str(q))) + str(q)
                    spx_str = 'd_' + zero_str + '_01'
                    spy_str = 'd_' + zero_str + '_02'
                    spz_str = 'd_' + zero_str + '_03'
                    Bx.append(A[spx_str])
                    By.append(A[spx_str])
                    Bz.append(A[spx_str])
                    x = imp[q - 1, 0]
                    y = imp[q - 1, 1]
                    r = np.sqrt(x**2 + y**2)
                    R.append(r)
                    z = imp[q - 1, 2]
                    Z.append(z)
                    phi = np.arctan2(x, y)
                    Phi.append(phi)
                    theta = np.arctan2(z, r)
                    that = [-np.sin(phi), np.cos(phi), 0]
                    rhat = [np.cos(phi), np.sin(phi), 0]
                    B_pol.append(np.sin(theta) * (rhat[0] * Bx[q - 1] + \
                        rhat[1] * By[q - 1]) + np.cos(theta) * Bz[q - 1])
                    B_tor.append(that[0] * Bx[q - 1] + that[1] * By[q - 1])
                    B_rad.append(rhat[0] * Bx[q - 1] + rhat[1] * By[q - 1])
                A['imp_r'] = R
                A['imp_z'] = Z
                A['imp_phi'] = Phi
                A['imp_Bx'] = Bx
                A['imp_By'] = By
                A['imp_Bz'] = Bz
                A['imp_Bpol'] = B_pol
                A['imp_Btor'] = B_tor
                A['imp_Brad'] = B_rad
                A['imp_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_inter_probes':
                inter_density = []
                for q in range(1, num_inter_probes + 1):
                    zero_str = '0' * (6 - len(str(q))) + str(q)
                    spx_str = 'd_' + zero_str + '_01'
                    spy_str = 'd_' + zero_str + '_02'
                    spz_str = 'd_' + zero_str + '_03'
                    inter_density.append(A[spx_str])
                A['inter_n'] = np.mean(inter_density,0)
                A['inter_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsn_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (nchords,npts,np.shape(A['B'])[1]))
                A['ids_n'] = A.pop('B')
                A['ids_ntime'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsT_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (nchords,npts,np.shape(A['B'])[1]))
                A['ids_T'] = A.pop('B')
                A['ids_ntime'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsV_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (3,nchords,npts,np.shape(A['B'])[1]))
                A['ids_V'] = A.pop('B')
                A['ids_ntime'] = A.pop('time')
            for k in range(len(psilist)):
                if psilist[k] in A:
                    p[psilist[k]] = A[psilist[k]]
    p['tcurr'] = p['tcurr'] / mu0
    p['curr01'] = p['curr01'] / mu0
    p['curr02'] = p['curr02'] / mu0
    p['is_HITSI3'] = False
    if is_HITSI3:
        p['curr03'] = p['curr03'] / mu0
        p['is_HITSI3'] = True
    p['filename'] = 'Psi-Tet' + str(f_1)
    p['dt'] = p['sp_time'][0,1]-p['sp_time'][0,0]
    if path.exists(runfolder + 'Psi-Tet' + str(f_1) + '_xmhd.mat'):
        flatten_object(p)
        interpolate_all(p)
        get_time_limits(p,limits)
    return p

## Loads a psi-tet mat file using the new format used in newer versions
# of psi-tet, including the 2T files
# @param fname Name of the base file
# @param runfolder Name of the folder where the file resides
# @param f_1 f_1 in kHz
# @param is_HITSI3 Flag to indicate if this is a HITSI3 run
# @returns psitet List of psi-tet dictionaries
def load_psitet_new(fname, runfolder, f_1, is_HITSI3, limits):
    num_sp_probes = 96
    num_imp_probes = 100
    num_inter_probes = 100
    sp = np.loadtxt(runfolder + 'sp_locations.txt', dtype='str')
    imp = np.loadtxt(runfolder + 'imp_locations.txt', dtype='float')
    thomson = np.loadtxt(runfolder + 'thomson_locations.txt', dtype='float')
    ids = np.loadtxt(runfolder + 'ids_locations.txt', dtype='float')
    sp_names = sp[:, 0]
    sp_R = sp[:, 1].astype(float)
    sp_Z = sp[:, 2].astype(float)
    sp_phi = sp[:, 3].astype(float)
    sp_theta = np.arctan2(sp_Z, sp_R - 33)
    B_theta = sp[:, 4].astype(float)
    # Load psi-tet 1T files
    mat_files = ['xmhd', 'hitn_driver', 'heat_flux', 'hitn_sp_probes', 'hitn_imp_probes',
                 'hitn_inter_probes', 'hitn_idsn_probes', 'hitn_idsT_probes', 'hitn_idsV_probes',
                 'hitn_thomson_probes']
    p = psiObject()
    # This list is alphabetically sorted
    psilist = [a for a in dir(p) if not a.startswith('__')]
    p = asdict(p)
    for j in range(len(mat_files)):
        filename = runfolder + fname + str(f_1) + \
            '_' + mat_files[j] + '.mat'
        print(filename)
        if path.exists(filename):
            A = sio.loadmat(filename)
            if mat_files[j] == 'hitn_driver':
                A['driver_time'] = A.pop('time')
            elif mat_files[j] == 'heat_flux':
                A['heat_flux_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_sp_probes':
                B = A['B']
                B = np.reshape(B, \
                    (np.shape(B)[0], int(np.shape(B)[1] / 3), 3))
                Bx = B[:, :, 0]
                By = B[:, :, 1]
                Bz = B[:, :, 2]
                A['sp_names'] = sp_names
                A['sp_r'] = sp_R
                A['sp_z'] = sp_Z
                A['sp_phi'] = sp_phi
                tpos = sp_phi * pi / 180
                pangle = B_theta * pi / 180
                that = [-np.sin(tpos), np.cos(tpos), 0]
                rhat = [np.cos(tpos), np.sin(tpos), 0]
                B_pol = np.sin(
                    pangle) * (rhat[0] * Bx + rhat[1] * By) + np.cos(pangle) * Bz
                B_tor = that[0] * Bx + that[1] * By
                A['sp_Bx'] = Bx
                A['sp_By'] = By
                A['sp_Bz'] = Bz
                A['sp_Bpol'] = np.transpose(B_pol)
                A['sp_Btor'] = np.transpose(B_tor)
                A['sp_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_imp_probes':
                B = A['B']
                B = np.reshape(B, \
                    (np.shape(B)[0], int(np.shape(B)[1] / 3), 3))
                Bx = B[:, :, 0]
                By = B[:, :, 1]
                Bz = B[:, :, 2]
                x = imp[:, 0]
                y = imp[:, 1]
                r = np.sqrt(x**2 + y**2)
                z = imp[:, 2]
                phi = np.arctan2(x, y)
                A['imp_r'] = r
                A['imp_z'] = z
                A['imp_phi'] = phi
                theta = np.arctan2(z, r)
                that = [-np.sin(phi), np.cos(phi), 0]
                rhat = [np.cos(phi), np.sin(phi), 0]
                B_pol = np.sin(
                    theta) * (rhat[0] * Bx + rhat[1] * By) + np.cos(theta) * Bz
                B_tor = that[0] * Bx + that[1] * By
                B_rad = rhat[0] * Bx + rhat[1] * By
                A['imp_Bx'] = Bx
                A['imp_By'] = By
                A['imp_Bz'] = Bz
                A['imp_Bpol'] = np.transpose(B_pol)
                A['imp_Btor'] = np.transpose(B_tor)
                A['imp_Brad'] = np.transpose(B_rad)
                A['imp_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_inter_probes':
                inter_density = A['B']
                A['inter_n'] = np.reshape(np.mean(inter_density,1), \
                    (1,len(np.mean(inter_density,1))))
                A['inter_time'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsn_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (nchords,npts,np.shape(A['B'])[1]))
                A['ids_n'] = A.pop('B')
                A['ids_ntime'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsT_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (nchords,npts,np.shape(A['B'])[1]))
                A['ids_T'] = A.pop('B')
                A['ids_Ttime'] = A.pop('time')
            elif mat_files[j] == 'hitn_idsV_probes':
                A['B'] = np.transpose(A['B'])
                A['B'] = np.reshape(A['B'], \
                    (3,nchords,npts,np.shape(A['B'])[1]))
                A['ids_V'] = A.pop('B')
                A['ids_Vtime'] = A.pop('time')
            elif mat_files[j] == 'hitn_thomson_probes':
                A['thomson_te'] = A.pop('B')
                A['thomson_time'] = A.pop('time')
            for k in range(len(psilist)):
                if psilist[k] in A:
                    p[psilist[k]] = A[psilist[k]]
    p['inj_power'] = p['iwall'] + p['ewall'] +\
        p['fpow'] + p['ppow'] + p['therm']
    p['tcurr'] = p['tcurr'] / mu0
    p['curr01'] = p['curr01'] / mu0
    p['curr02'] = p['curr02'] / mu0
    p['is_HITSI3'] = False
    if is_HITSI3:
        p['curr03'] = p['curr03'] / mu0
        p['is_HITSI3'] = True
    p['filename'] = 'Psi-Tet-2T' + str(f_1)
    p['dt'] = p['sp_time'][0,1]-p['sp_time'][0,0]
    if path.exists(runfolder + 'Psi-Tet-2T' + str(f_1) + '_xmhd.mat'):
        flatten_object(p)
        interpolate_all(p)
        get_time_limits(p,limits)
    return p

## Flattens any keys in the dictionary which are of sizes
## like (3,1,400) to (3,400) for ease of processing later
# @param dict A psi-tet dictionary
def flatten_object(dict):
    for key in dict.keys():
        if len(np.shape(dict[key])) == 2 and \
                (np.shape(dict[key])[0] == 1 or \
                np.shape(dict[key])[1] == 1):
            dict[key] = np.ravel(dict[key])
        if len(np.shape(dict[key])) == 3 and \
                np.shape(dict[key])[0] == 1:
            dict[key] = np.reshape(dict[key], \
                (np.shape(dict[key])[1],np.shape(dict[key])[2]))
        if len(np.shape(dict[key])) == 3 and \
                np.shape(dict[key])[1] == 1:
            dict[key] = np.reshape(dict[key], \
                (np.shape(dict[key])[0],np.shape(dict[key])[2]))

## Function which sets the time limits of the data
# @param dict A psi-tet dictionary
# @param limits An array of two floats indicating the analysis window
def get_time_limits(dict,limits):
    idx1 = (np.abs(dict['sp_time'] - limits[0]*1e-3)).argmin()
    idx2 = (np.abs(dict['sp_time'] - limits[1]*1e-3)).argmin()
    dict['t0'] = idx1
    dict['tf'] = idx2

## Loads a single file into a psi-tet dictionary
# @param filename Full path name of the file
# @param f_1 Injector Frequency in kHz
# @param is_psitet Flag to check if this is a psi-tet file
# @param is_2T Flag to check if this is a psi-tet 2T file
# @param is_HITSI3 Flag to indicate this is a HITSI3 run
# @returns p A psi-tet dictionary
def loadshot(filename, directory, f_1, is_psitet, is_2T, \
    is_HITSI3, limits):
    if is_psitet and is_2T:
        p = load_psitet_new(filename,directory,f_1,is_HITSI3,limits)
    elif is_psitet:
        p = load_psitet_old(filename,directory,f_1,is_HITSI3,limits)
    else:
        p = asdict(psiObject())
        if path.exists(directory+filename):
            sio.loadmat(directory+filename,mdict=p)
            p['sp_time'] = p['time']
            if is_HITSI3:
                p['is_HITSI3'] = True
            else:
                p['is_HITSI3'] = False
            p['filename'] = filename
            p['dt'] = p['time'][0,1]-p['time'][0,0]
            flatten_object(p)
            get_time_limits(p,limits)
    return p

## Interpolate all signals to the surface probe time base
# @param dict A psi-tet dictionary
def interpolate_all(dict):
    time = dict['time']
    driver_time = dict['driver_time']
    inter_time = [0]
    imp_time = [0]
    ids_ntime = [0]
    ids_Ttime = [0]
    ids_Vtime = [0]
    if 'inter_time' in dict.keys():
        inter_time = dict['inter_time']
    if 'imp_time' in dict.keys():
        imp_time = dict['imp_time']
    if 'ids_ntime' in dict.keys():
        ids_ntime = dict['ids_ntime']
        ids_Ttime = dict['ids_Ttime']
        ids_Vtime = dict['ids_Vtime']
    sp_time = dict['sp_time']
    for key in dict.keys():
        if len(sp_time) in np.shape(dict[key]):
            continue
        elif len(time) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,time,dict[key])
        elif len(driver_time) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,driver_time,dict[key])
        elif len(inter_time) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,inter_time,dict[key])
        elif len(imp_time) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,imp_time,dict[key])
        elif len(ids_ntime) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,ids_ntime,dict[key])
        elif len(ids_Ttime) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,ids_Ttime,dict[key])
        elif len(ids_Vtime) in np.shape(dict[key]):
            dict[key] = np.interp(sp_time,ids_Vtime,dict[key])
