## @package utilities
## Defines various functions for smoothing, calculating
## fourier transforms, SVD, and so on.
from plot_attributes import *
from map_probes import \
    sp_name_dict, dead_probes, \
    imp_phis8, imp_phis32, midphi, \
    imp_rads, Z_bighit, R_bighit, Phi
from scipy.stats import linregress
import csv

## Python equivalent of the sihi_smooth function found in older
## matlab scripts. This does a boxcar average.
## Code simplified to only work for real-valued signals.
# @param y Signal in time
# @param time Time base associated with y
# @param f_1 Injector Frequency with which to apply the smoothing
# @returns x Smoothed signal in time
def sihi_smooth(y, time, f_1):
    injCyc = 1.0 / (1000.0 * f_1)
    Navg = 100
    Navg2 = int(Navg / 2.0)
    # make it 100 time points per injector cycle
    tint = np.linspace(time[0], time[len(time) - 1],
        int((time[len(time) - 1]-time[0]) / (injCyc / Navg)))
    yint = np.interp(tint, time, y)
    xint = np.zeros(len(tint))
    xint[0:Navg2] = np.mean(yint[0:Navg])

    for it in range(Navg2,len(tint) - Navg2):
        xint[it] = xint[it - 1] + \
            (yint[it + Navg2] - yint[it - Navg2]) / Navg

    xint[0:Navg2] = xint[Navg2]
    xint[len(tint) - Navg2:len(tint)] = \
        xint[len(tint) - Navg2 - 1]
    x = np.interp(time, tint, xint)
    x[len(x) - 1] = x[len(x) - 2]
    x[np.asarray(np.isnan(x)).nonzero()] = 0
    return x

## Performs a SVD of the data in a psi-tet dictionary.
## Has dmd_flags to control which data is put into the matrix
## for the SVD.
# @param psi_dict A psi-tet dictionary
def SVD(psi_dict):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    data = np.vstack((psi_dict['curr01'],psi_dict['curr02']))
    if psi_dict['is_HITSI3'] == True:
        data = np.vstack((data,psi_dict['curr03']))
    #data = np.vstack((data,psi_dict['flux01']))
    #data = np.vstack((data,psi_dict['flux02']))
    data = np.vstack((data,psi_dict['sp_Bpol']))
    data = np.vstack((data,psi_dict['sp_Btor']))
    getshape = np.shape(data)[0]
    if psi_dict['use_IMP']:
        if psi_dict['num_IMPs'] == 8:
            skip = 40
            psi_dict['imp_Bpol'] = np.nan_to_num(psi_dict['imp_Bpol'])[::skip,:]
            psi_dict['imp_Btor'] = np.nan_to_num(psi_dict['imp_Btor'])[::skip,:]
            psi_dict['imp_Brad'] = np.nan_to_num(psi_dict['imp_Brad'])[::skip,:]
            bindices = slice(0,29,4)
            indices = list(range(0,32))
            del indices[bindices]
            psi_dict['imp_Bpol'] = psi_dict['imp_Bpol'][indices,:]
            psi_dict['imp_Btor'] = psi_dict['imp_Btor'][indices,:]
            psi_dict['imp_Brad'] = psi_dict['imp_Brad'][indices,:]
        if psi_dict['num_IMPs'] == 32:
            skip = 1
            psi_dict['imp_Bpol'] = np.nan_to_num(psi_dict['imp_Bpol'])[::skip,:]
            psi_dict['imp_Btor'] = np.nan_to_num(psi_dict['imp_Btor'])[::skip,:]
            psi_dict['imp_Brad'] = np.nan_to_num(psi_dict['imp_Brad'])[::skip,:]
        data = np.vstack((data,psi_dict['imp_Bpol']))
        shape1 = np.shape(psi_dict['imp_Bpol'])[0]
        shape2 = np.shape(psi_dict['imp_Btor'])[0]
        shape3 = np.shape(psi_dict['imp_Brad'])[0]
        imp_pol_indices = np.linspace(0,shape1,shape1, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Btor']))
        imp_tor_indices = np.linspace(shape1,shape2+shape1,shape2, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Brad']))
        imp_rad_indices = np.linspace(shape1+shape2, \
            shape3+shape2+shape1,shape3, \
            dtype = 'int')

    # correct injector currents
    if psi_dict['is_HITSI3'] == True:
        data[0:3,:] = data[0:3,:]*mu0
    else:
        data[0:2,:] = data[0:2,:]*mu0
    # For forecasting
    #psi_dict['full_data'] = data[:,t0:tf]
    #data = data[:,t0:t0+int(tf/2)-1]
    data = data[:,t0:tf]
    noise = np.random.normal(0,5e-4,(np.shape(data)[0],np.shape(data)[1]))
    data_sub = -data #+noise
    #data_sub = subtract_linear_trend(psi_dict,data)
    u,s,v = np.linalg.svd(data_sub)
    v = np.conj(np.transpose(v))
    psi_dict['SVD_data'] = data_sub
    psi_dict['SP_data'] = data
    psi_dict['U'] = u
    psi_dict['S'] = s
    psi_dict['V'] = v

## Identifies and subtracts a linear trend from each
## of the time signals contained in
## the SVD data associated with the dictionary 'psi_dict'. This is
## to help DMD algorithms, since DMD does not deal well with
## non-exponential growth.
# @param psi_dict A dictionary with SVD data
# @param data The SVD data matrix
# @returns data_subtracted The SVD data matrix
#  with the linear trend subtracted off
def subtract_linear_trend(psi_dict,data):
    state_size = np.shape(data)[0]
    tsize = np.shape(data)[1]
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]
    dt = psi_dict['sp_time'][1] - psi_dict['sp_time'][0]
    data_subtracted = np.zeros((state_size,tsize))
    for i in range(state_size):
        slope, intercept, r_value, p_value, std_err = linregress(time,data[i,:])
        data_subtracted[i,:] = data[i,:] - (slope*time+intercept)
        if i == 10:
            plt.figure()
            plt.plot(time,data_subtracted[i,:],'g')
            plt.plot(time,slope*time+intercept,'b')
            plt.plot(time,data[i,:],'r')
            plt.savefig(out_dir+'linear_trend_test.png')
    return data_subtracted

## Performs the fourier calculation based on Wrobel 2011
# @param nmax The toroidal/poloidal number resolution
# @param tsize The number of time snapshots
# @param b Magnetic field signals of a toroidal/poloidal set of probes
# @param phi Toroidal/poloidal angles associated with the probes
# @returns amps The toroidal/poloidal mode amplitudes
def fourier_calc(nmax,tsize,b,phi):
    # Set up mode calculation- code adapted from JSW
    minvar = 1e-10 # minimum field variance for calcs.
    amps = np.zeros((nmax+1,tsize))
    phases = np.zeros((nmax+1,tsize))
    vardata = np.zeros(np.shape(b)) + minvar
        # Calculate highest nmax possible
    nprobes = np.shape(b)[0]
    mcoeff = np.zeros((2*nmax + 1,2*nmax + 1))
    for nn in range(nmax+1):
        for m in range(nmax+1):
            mcoeff[m, nn] = \
                sum(np.cos(m*phi) * np.cos(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax, nn] = \
                sum(np.sin(m*phi) * np.cos(nn*phi))

    for nn in range(1,nmax+1):
        for m in range(nmax+1):
            mcoeff[m,nn+nmax] = \
                sum(np.cos(m*phi) * np.sin(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax,nn+nmax] = \
                sum(np.sin(m*phi) * np.sin(nn*phi))

    asnbs    = np.zeros(2*nmax + 1)
    varasnbs = np.zeros(2*nmax + 1)
    rhs      = np.zeros(2*nmax + 1)
    veca     = np.zeros((nmax+1,tsize))
    vecb     = np.zeros((nmax,tsize))
    for m in range(tsize):
        bflds = b[:, m]
        varbflds = vardata[:, m]
        for nn in range(nmax+1):
            rhs[nn] = sum(bflds*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(bflds*np.sin(nn*phi))
        asnbs,g1,g2,g3 = np.linalg.lstsq(mcoeff,rhs)
        for nn in range(nmax+1):
            rhs[nn] = sum(np.sqrt(varbflds)*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(np.sqrt(varbflds)*np.sin(nn*phi))
        veca[0:nmax+1, m] = asnbs[0:nmax+1]
        vecb[0:nmax, m] = asnbs[nmax+1:2*nmax+1]
    amps[0,:] = veca[0, :]
    phases[0,:] = 0.0 * veca[0, :]

    for m in range(nmax):
        amps[m+1,:] = \
            np.sqrt(veca[m+1, :]**2 + vecb[m, :]**2)
        phases[m+1,:] = np.arctan2(vecb[m, :], veca[m+1, :])
    return amps

## Plots the toroidal current for a shot
# @param psi_dict A psi-tet dictionary
def plot_itor(psi_dict):
    itor = psi_dict['tcurr']/1000.0
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['time']*1000.0
    plt.figure(75000,figsize=(figx, figy))
    plt.plot(time,itor,color='orange',linewidth=lw)
    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='upper right')
    plt.axvline(x=time[t0],color='k')
    plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
   # plt.ylabel(r'$I_{tor}$ (kA)', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_current.png')
    plt.savefig(out_dir+'toroidal_current.eps')
    plt.savefig(out_dir+'toroidal_current.pdf')
    plt.savefig(out_dir+'toroidal_current.svg')

## Plots the BD chronos for a shot
# @param psi_dict A psi-tet dictionary
def plot_chronos(psi_dict):
    Vh = np.transpose(np.conj(psi_dict['V']))
    S = psi_dict['S']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(85000,figsize=(figx, figy))
    for i in range(3):
        plt.plot(time,S[i]*Vh[i,:]*1e4/S[0],'m',linewidth=lw, alpha=alphas[i], \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()],label='Mode '+str(i+1))
    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,loc='lower left')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Time (ms)', fontsize=fs)
    #h =# plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}V_{ki}^*$', fontsize=fs)
    #h.set_rotation(0)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD_chronos.png')
    plt.savefig(out_dir+'BD_chronos.eps')
    plt.savefig(out_dir+'BD_chronos.pdf')
    plt.savefig(out_dir+'BD_chronos.svg')
    plt.figure(95000,figsize=(figx, figy))
    plt.semilogy(range(1,len(S)+1),S/S[0],'mo',markersize=ms,markeredgecolor='k')
    plt.semilogy(range(1,len(S)+1),S/S[0],'m')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    #plt.xlabel('Mode Number k', fontsize=fs)
    #h =# plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}$', fontsize=fs)
    #h.set_rotation(0)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD.png')
    plt.savefig(out_dir+'BD.eps')
    plt.savefig(out_dir+'BD.pdf')
    plt.savefig(out_dir+'BD.svg')

## Write out X,Y,Z,Bx,By,Bz into a CSV file (for plotly, perhaps)
# @param psi_dict A psi-tet dictionary
def write_Bfield_csv(psi_dict):
    size_bpol = np.shape(psi_dict['sp_Bpol'])[0]
    size_btor = np.shape(psi_dict['sp_Btor'])[0]
    size_imp_bpol = np.shape(psi_dict['imp_Bpol'])[0]
    size_imp_btor = np.shape(psi_dict['imp_Btor'])[0]
    size_imp_brad = np.shape(psi_dict['imp_Brad'])[0]
    tstep = 500
    offset = 2
    Bdata = np.reshape(psi_dict['SVD_data'][:,tstep], \
        (np.shape(psi_dict['SVD_data'])[0],1))
    Bz = np.vstack((Bdata[offset:offset+size_bpol], \
        Bdata[offset+size_bpol+size_btor: \
        offset+size_bpol+size_btor+size_imp_bpol]))
    Btor = np.vstack((Bdata[offset+size_bpol: \
        offset+size_bpol+size_btor], \
        Bdata[offset+size_bpol+size_btor+size_imp_bpol: \
        offset+size_bpol+size_btor+size_imp_bpol+size_imp_btor]))
    Brad = np.vstack((np.zeros((size_bpol,1)), \
        Bdata[offset+size_bpol+size_btor+size_imp_bpol+size_imp_btor: \
        offset+size_bpol+size_btor+size_imp_bpol+size_imp_btor+size_imp_brad]))
    zprobes = np.zeros(np.shape(Bz)[0])
    rprobes = np.zeros(np.shape(Bz)[0])
    phiprobes = np.zeros(np.shape(Bz)[0])
    num_IMPs = psi_dict['num_IMPs']
    phis_imp = np.zeros(160*num_IMPs)
    rads_imp = np.zeros(160*num_IMPs)
    for i in range(num_IMPs):
        if num_IMPs == 8:
          phis_imp[i*160:(i+1)*160] = np.ones(160)*imp_phis8[i]
          skip = 40
        elif num_IMPs == 32:
          phis_imp[i*160:(i+1)*160] = np.ones(160)*imp_phis32[i]
          skip = 1
        else:
          print('Invalid number of IMPs, exiting')
          exit()
        rads_imp[i*160:(i+1)*160] = np.ones(160)*imp_rads
    q = 0
    qq = 0
    for i in range(96):
        if i < 64:
            zprobes[i] = Z_bighit[q]
            rprobes[i] = R_bighit[q]
            phiprobes[i] = Phi[i % 4]
            if (i+1) % 4 == 0:
                q = q + 1
        else:
            zprobes[i] = Z_bighit[16+(i%2)]
            rprobes[i] = R_bighit[16+(i%2)]
            phiprobes[i] = midphi[qq]
            if (i+1) % 2 == 0:
                qq = qq + 1
    zprobes[96:] = 0.0
    rprobes[96:] = rads_imp
    phiprobes[96:] = phis_imp
    zprobes = zprobes.flatten()
    rprobes = rprobes.flatten()
    phiprobes = phiprobes.flatten()
    Bz = Bz.flatten()
    Brad = Brad.flatten()
    Btor = Btor.flatten()
    xprobes = rprobes*np.cos(phiprobes)
    yprobes = rprobes*np.sin(phiprobes)
    Bx = Brad*np.cos(phiprobes) - Btor*np.sin(phiprobes)
    By = Brad*np.sin(phiprobes) + Btor*np.cos(phiprobes)
    saveArray = [xprobes,yprobes,zprobes,Bx,By,Bz]
    np.savetxt('plotly_data.csv',np.transpose(saveArray),delimiter=',')

## Makes the bar plots from the dmdpaper for each of the coherent
## modes and each of the DMD methods
# @param psi_dict A psi-tet dictionary
def bar_plot(psi_dict):
    f_1 = psi_dict['f_1']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    t_vec = psi_dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(psi_dict['sp_Bpol'])[0]
    size_btor = np.shape(psi_dict['sp_Btor'])[0]
    size_imp_bpol = np.shape(psi_dict['imp_Bpol'])[0]
    size_imp_btor = np.shape(psi_dict['imp_Btor'])[0]
    size_imp_brad = np.shape(psi_dict['imp_Brad'])[0]
    offset = 2
    if psi_dict['is_HITSI3'] == True:
        offset = 3
    tsize = len(t_vec)
    num_IMPs = psi_dict['num_IMPs']
    nmax = 7
    phis = np.zeros(160*num_IMPs)
    if num_IMPs == 8:
        imp_phis = imp_phis8
        nmax_imp = 3
        skip = 40
    elif num_IMPs == 32:
        imp_phis = imp_phis32
        nmax_imp = 10
        skip = 1
    else:
        print('Invalid number for the number of IMPs')
        exit()
    for i in range(num_IMPs):
        phis[i*160:(i+1)*160] = np.ones(160)*imp_phis[i]
    # subsample as needed
    phis = phis[::skip]
    orig_phis = phis[:len(phis)]
    amps_total = []
    for i in range(3):
        if i == 0:
            amp_indices = np.arange(offset+size_bpol+size_btor, \
                offset+size_bpol+size_btor+size_imp_bpol)
        if i == 1:
            amp_indices = np.arange(offset+size_bpol-32, \
                offset+size_bpol,2)
        if i == 2:
            amp_indices = np.arange(offset,offset+size_bpol)

        Bfield_f0 = psi_dict['sparse_Bfield_f0'][amp_indices,:]
        Bfield_f1 = psi_dict['sparse_Bfield_f1'][amp_indices,:]
        Bfield_f2 = psi_dict['sparse_Bfield_f2'][amp_indices,:]
        Bfield_f3 = psi_dict['sparse_Bfield_f3'][amp_indices,:]
        Bfield_kink = psi_dict['optimized_Bfield_kink'][amp_indices,:]
        if i == 0:
            amps = np.zeros((nmax_imp+1,160,tsize,5))
            if num_IMPs == 8:
                bindices = slice(0,29,4)
                indices = list(range(0,32))
                del indices[bindices]
                phis = orig_phis[indices]
                for k in range(3):
                    amps[:,k,:,0] = fourier_calc(nmax_imp,tsize,Bfield_f0[k::3,:],phis[k::3])
                    amps[:,k,:,1] = fourier_calc(nmax_imp,tsize,Bfield_f1[k::3,:],phis[k::3])
                    amps[:,k,:,2] = fourier_calc(nmax_imp,tsize,Bfield_f2[k::3,:],phis[k::3])
                    amps[:,k,:,3] = fourier_calc(nmax_imp,tsize,Bfield_f3[k::3,:],phis[k::3])
                    amps[:,k,:,4] = fourier_calc(nmax_imp,tsize,Bfield_kink[k::3,:],phis[k::3])
                avg_amps = np.mean(abs(amps[:,0:4,:]),axis=1)
            elif num_IMPs == 32:
                for k in range(160):
                    print(np.shape(Bfield_f0[k::160,:]),np.shape(phis[k::160]))
                    amps[:,k,:,0] = fourier_calc(nmax_imp,tsize,Bfield_f0[k::160,:],phis[k::160])
                    amps[:,k,:,1] = fourier_calc(nmax_imp,tsize,Bfield_f1[k::160,:],phis[k::160])
                    amps[:,k,:,2] = fourier_calc(nmax_imp,tsize,Bfield_f2[k::160,:],phis[k::160])
                    amps[:,k,:,3] = fourier_calc(nmax_imp,tsize,Bfield_f3[k::160,:],phis[k::160])
                    amps[:,k,:,4] = fourier_calc(nmax_imp,tsize,Bfield_kink[k::160,:],phis[k::160])
                avg_amps = np.mean(abs(amps),axis=1)
        if i == 1:
            amps = np.zeros((nmax+1,tsize,5))
            phi = midphi
            amps[:,:,0] = fourier_calc(nmax,tsize,Bfield_f0,phi)
            amps[:,:,1] = fourier_calc(nmax,tsize,Bfield_f1,phi)
            amps[:,:,2] = fourier_calc(nmax,tsize,Bfield_f2,phi)
            amps[:,:,3] = fourier_calc(nmax,tsize,Bfield_f3,phi)
            amps[:,:,4] = fourier_calc(nmax,tsize,Bfield_kink,phi)
            avg_amps = abs(amps)
        if i == 2:
            amps = np.zeros((nmax+1,4,tsize,5))
            # Find the poloidal gap probes
            k1 = 0
            k2 = 0
            j = 0
            theta = np.zeros(16)
            temp_B_f0 = np.zeros((64,tsize))
            temp_B_f1 = np.zeros((64,tsize))
            temp_B_f2 = np.zeros((64,tsize))
            temp_B_f3 = np.zeros((64,tsize))
            temp_B_kink = np.zeros((64,tsize))
            temp_theta = np.zeros(16)
            for key in sp_name_dict.keys():
                if key in dead_probes:
                    if key[5] == 'P':
                        k2 = k2 + 1
                    continue
                if key[5] == 'P' and \
                    key[2:5] != 'L05' and key[2:5] != 'L06':
                    temp_B_f0[k2, :] = Bfield_f0[j, :tsize]
                    temp_B_f1[k2, :] = Bfield_f1[j, :tsize]
                    temp_B_f2[k2, :] = Bfield_f2[j, :tsize]
                    temp_B_f3[k2, :] = Bfield_f3[j, :tsize]
                    temp_B_kink[k2, :] = Bfield_kink[j, :tsize]
                if key[5:9] == 'P225' and \
                    key[2:5] != 'L05' and key[2:5] != 'L06':
                    temp_theta[k1] = sp_name_dict[key][3]
                    k1 = k1 + 1
                if key[5] == 'P':
                    j = j + 1
                    k2 = k2 + 1
                phi_str = [r'$0^o$',r'$45^o$', \
                    r'$180^0$',r'$225^o$']
            for j in range(4):
                B_f0 = temp_B_f0[j::4,:]
                B_f1 = temp_B_f1[j::4,:]
                B_f2 = temp_B_f2[j::4,:]
                B_f3 = temp_B_f3[j::4,:]
                B_kink = temp_B_kink[j::4,:]
                inds = ~np.all(B_f0 == 0, axis=1)
                B_f0 = B_f0[inds]
                B_f1 = B_f1[inds]
                B_f2 = B_f2[inds]
                B_f3 = B_f3[inds]
                B_kink = B_kink[inds]
                theta = temp_theta[np.where(inds)]
                amps[:,j,:,0] = fourier_calc(nmax,tsize,B_f0,theta)
                amps[:,j,:,1] = fourier_calc(nmax,tsize,B_f1,theta)
                amps[:,j,:,2] = fourier_calc(nmax,tsize,B_f2,theta)
                amps[:,j,:,3] = fourier_calc(nmax,tsize,B_f3,theta)
                amps[:,j,:,4] = fourier_calc(nmax,tsize,B_kink,theta)
            avg_amps = np.mean(abs(amps),axis=1)

        amps_total.append(avg_amps)
        plt.figure(768,figsize=(figx, figy))
        plt.subplot(1,3,i+1)
        width = 0.1
        alphas = np.linspace(1.0,0.05,11)
        ax = plt.gca()
        ax.set_axisbelow(True)
        if i > 0:
            for j in range(nmax+1):
                if (j > 3 and nmax_imp == 3) or (j < 4 and nmax_imp == 10):
                    if i < 2:
                        plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                            alpha=alphas[j],edgecolor='k')
                    if i == 2:
                        plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                            alpha=alphas[j],edgecolor='k')
                else:
                    if i < 2:
                        plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                            label='n = '+str(j),alpha=alphas[j],edgecolor='k')
                    if i == 2:
                        plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                            label='m = '+str(j),alpha=alphas[j],edgecolor='k')
            plt.legend(edgecolor='k',facecolor='white',framealpha=1,fontsize=20,loc='upper right')
            ax.set_xticks(np.arange(4)+7*width/2.0)
        else:
            nm = min(nmax,nmax_imp)
            for j in range(nm+1): #nmax_imp if want all the modes
                if j < 4 and nmax_imp != 3:
                    plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                        alpha=alphas[j],edgecolor='k')
                else:
                    plt.bar(np.arange(4)+width*j,avg_amps[j,0,0:4]*1e4,width, \
                        label='n = '+str(j),alpha=alphas[j],edgecolor='k')
            plt.legend(edgecolor='k',facecolor='white',framealpha=1,fontsize=20,loc='upper right')
            if nmax_imp != 3:
                ax.set_xticks(np.arange(4)+7*width/2.0)
            else:
                ax.set_xticks(np.arange(4)+3*width/2.0)
        plt.yscale('log')
        plt.ylim(1e-2,1e3)
        plt.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
     #   ax.set_xticks([0,1,2,3])
        ax.set_yticks([1e-2,1e-1,1e0,1e1,1e2,1e3])
        if i != 0:
            ax.set_yticklabels([])
            #ax.set_yticklabels([])
        ax.set_xticklabels([r'$f_0$',r'$f_1^{inj}$',r'$f_2^{inj}$',r'$f_3^{inj}$','',''])

        plt.figure(769,figsize=(figx, figy))
        plt.subplot(1,3,i+1)
        width = 0.1
        alphas = np.linspace(1.0,0.05,11)
        ax = plt.gca()
        ax.set_axisbelow(True)
        if i > 0:
            for j in range(nmax+1):
                if (j > 3 and nmax_imp == 3) or (j < 4 and nmax_imp == 10):
                    if i < 2:
                        plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                            alpha=alphas[j],edgecolor='k')
                    if i == 2:
                        plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                            alpha=alphas[j],edgecolor='k')
                else:
                    if i < 2:
                        plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                            label='n = '+str(j),alpha=alphas[j],edgecolor='k')
                    if i == 2:
                        plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                            label='m = '+str(j),alpha=alphas[j],edgecolor='k')
            plt.legend(edgecolor='k',facecolor='white',framealpha=1,fontsize=20,loc='upper right')
            ax.set_xticks(np.arange(1)+7*width/2.0)
        else:
            nm = min(nmax,nmax_imp)
            for j in range(nm+1): #nmax_imp if want all the modes
                if j < 4 and nmax_imp != 3:
                    plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                        alpha=alphas[j],edgecolor='k')
                else:
                    plt.bar(np.arange(1)+width*j,avg_amps[j,tsize-2,4]*1e4,width, \
                        label='n = '+str(j),alpha=alphas[j],edgecolor='k')
            plt.legend(edgecolor='k',facecolor='white',framealpha=1,fontsize=20,loc='upper right')
            if nmax_imp != 3:
                ax.set_xticks(np.arange(1)+7*width/2.0)
            else:
                ax.set_xticks(np.arange(1)+3*width/2.0)
        plt.yscale('log')
        plt.ylim(1e-2,1e3)
        plt.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        #   ax.set_xticks([0,1,2,3])
        ax.set_yticks([1e-2,1e-1,1e0,1e1,1e2,1e3])
        if i != 0:
            ax.set_yticklabels([])
            #ax.set_yticklabels([])
        ax.set_xticklabels([r'$f_{kink}$'])

    plt.figure(768)
    plt.savefig(out_dir+'bars.pdf')
    plt.savefig(out_dir+'bars.png')
    plt.savefig(out_dir+'bars.eps')
    plt.figure(769)
    plt.savefig(out_dir+'bars_kinks.pdf')
    plt.savefig(out_dir+'bars_kinks.png')
    plt.savefig(out_dir+'bars_kinks.eps')
