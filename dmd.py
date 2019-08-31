## @package dmd
## This file contains the different dmd methods
from plot_attributes import *
from dmd_utilities import power_spectrum, freq_phase_plot
from numba import jit,config
import time as Clock

## DMD with sliding window
# @param total A list of psi-tet dictionaries
# @param inj_frequencies Injector frequencies of the dictionaries
# @param numwindows Number of windows for the sliding window
def DMD_slide(total,inj_frequencies,numwindows):
    fignum = len(total)*numwindows
    for k in range(len(total)):
        dict = total[k]
        inj_freq = inj_frequencies[k]
        dictname = dict['filename']
        t0 = dict['t0']
        tf = dict['tf']
        data = np.copy(dict['SVD_data'])
        time = dict['sp_time'][t0:tf]
        dt = dict['sp_time'][1] - dict['sp_time'][0]
        r = np.shape(data)[0]
        tsize = np.shape(data)[1]
        windowsize = int(np.floor(tsize/float(numwindows)))
        if numwindows==1:
            windowsize=windowsize-1
        if tsize >= windowsize:
            starts = np.linspace(0, \
                int(np.floor(tsize/float(numwindows)))*(numwindows-1),numwindows, dtype='int')
            ends = starts + np.ones(numwindows,dtype='int')*windowsize
        else:
            print('windowsize > tsize, dmd invalid')
        trunc = dict['trunc'] 
        dmd_b = np.zeros((r,tsize),dtype='complex')
        b_inj = np.zeros((r,tsize),dtype='complex')
        b_eq = np.zeros((r,tsize),dtype='complex')
        dict['dmd_y0'] = []
        dict['dmd_omega'] = []
        dict['dmd_phi'] = []
        S = np.zeros((trunc,trunc))
        for i in range(numwindows):
            X = data[:,starts[i]:ends[i]]
            Xprime = data[:,1+starts[i]:ends[i]+1]
            Udmd,Sdmd,Vdmd = np.linalg.svd(X,full_matrices=False)
            Vdmd = np.transpose(Vdmd)
            Udmd = Udmd[:,0:trunc]
            Sdmd = Sdmd[0:trunc]
            Vdmd = Vdmd[:,0:trunc]
            A = np.dot(np.dot(np.transpose(Udmd),Xprime),Vdmd/Sdmd)
            eigvals,Y = np.linalg.eig(A)
            Phi = np.dot(np.dot(Xprime,Vdmd/Sdmd),Y)
            omega = np.log(eigvals)/dt
            omega[np.isnan(omega).nonzero()] = 0
            decay = abs(np.real(omega)/(pi*2))
            oscillation = abs(np.imag(omega)/(pi*2))
            u_modes = np.zeros((trunc,windowsize),'complex')
            p_modes = np.zeros((trunc,windowsize),'complex')
            Vandermonde = np.zeros((trunc,windowsize),'complex')
            for iter in range(windowsize):
                u_modes[:,iter] = np.exp((omega-np.real(omega))* \
                    (time[starts[i]+iter]-time[starts[i]]))
                p_modes[:,iter] = np.exp(np.real(omega)* \
                    (time[starts[i]+iter]-time[starts[i]]))
                Vandermonde[:,iter] = u_modes[:,iter]*p_modes[:,iter]
            for jj in range(trunc):
                for kk in range(trunc):
                    if jj == kk:
                        S[jj,kk] = Sdmd[jj]
            q = np.conj(np.diag(np.dot(np.dot(np.dot( \
                Vandermonde,Vdmd),np.conj(S)),Y)))
            P = np.dot(np.conj(np.transpose(Y)),Y)* \
                np.conj(np.dot(Vandermonde, \
                np.conj(np.transpose(Vandermonde))))
            y0 = np.dot(np.linalg.inv(P),q)
            dict['dmd_y0'].append(y0)
            dict['dmd_omega'].append(omega)
            dict['dmd_phi'] = Phi
            typename = 'DMD'
            filename = 'power_DMD'+str(dictname)+'_'+str(i)+'.png'
            power_spectrum(y0,omega,inj_freq,filename,typename)
            filename = 'phasePlot_DMD'+str(dictname)+'_'+str(i)+'.png'
            freq_phase_plot(y0,omega,inj_freq,filename,typename)
            equilIndex = np.asarray(np.asarray(abs(np.imag(omega))==0).nonzero())
            if equilIndex.size==0:
                equilIndex = np.atleast_1d(np.argmin(abs(np.imag(omega))))
            equilIndex = np.ravel(equilIndex).tolist()
            injIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
                abs(np.imag(omega)/(2*pi)),inj_freq*1000.0,atol=1000)).nonzero()))
            for iter in range(windowsize):
                u_modes[:,iter] = y0*np.exp((omega-np.real(omega))* \
                    (time[starts[i]+iter]-time[starts[i]]))
                p_modes[:,iter] = np.exp(np.real(omega)* \
                    (time[starts[i]+iter]-time[starts[i]]))
                Vandermonde[:,iter] = u_modes[:,iter]*p_modes[:,iter]
            mag_energy = abs((u_modes*p_modes)*np.conj(u_modes*p_modes) \
                /np.sum((u_modes*p_modes)*np.conj(u_modes*p_modes)))
            energy_order = np.flip(np.argsort(np.mean(mag_energy,axis=1)))
            dmd_b[:,starts[i]:ends[i]] = 0.5*np.dot(Phi,u_modes*p_modes)
            for j in range(len(equilIndex)):
                e_ind = equilIndex[j]
                b_eq[:,starts[i]:ends[i]] += 0.5*np.outer(Phi[:,e_ind], \
                    u_modes[e_ind,:]*p_modes[e_ind,:])
            for j in range(len(injIndex)):
                i_ind = injIndex[j]
                b_inj[:,starts[i]:ends[i]] += 0.5*np.outer(Phi[:,i_ind], \
                    u_modes[i_ind,:]*p_modes[i_ind,:])
            dmd_b[:,starts[i]:ends[i]] += \
                np.conj(dmd_b[:,starts[i]:ends[i]])
            b_inj[:,starts[i]:ends[i]] += \
                np.conj(b_inj[:,starts[i]:ends[i]])
            b_eq[:,starts[i]:ends[i]] += \
                np.conj(b_eq[:,starts[i]:ends[i]])
            err = np.linalg.norm(X-dmd_b[:,starts[i]:ends[i]],'fro') \
                /np.linalg.norm(X,'fro')
            print(err)

        dict['dmd_b'] = dmd_b
        dict['dmd_b_eq'] = b_eq
        dict['dmd_b_inj'] = b_inj

## sparsity-promoting DMD with sliding window
# @param total A list of psi-tet dictionaries
# @param inj_frequencies Injector frequencies of the dictionaries
# @param numwindows Number of windows for the sliding window
def sparse_DMD_slide(total,inj_frequencies,numwindows):
    fignum = len(total)*numwindows
    for k in range(len(total)):
        dict = total[k]
        inj_freq = inj_frequencies[k]
        dictname = dict['filename']
        t0 = dict['t0']
        tf = dict['tf']
        size_bpol = np.shape(dict['sp_Bpol'])[0]
        size_btor = np.shape(dict['sp_Btor'])[0]
        size_imp_bpol = np.shape(dict['imp_Bpol'])[0]
        size_imp_btor = np.shape(dict['imp_Btor'])[0]
        size_imp_brad = np.shape(dict['imp_Brad'])[0]
        offset = 2
        if dict['is_HITSI3'] == True:
            offset = 3
        data = dict['SVD_data'] #\
            #[offset+size_bpol+size_btor: \
            #offset+size_bpol+size_btor+size_imp_bpol,:]
        time = dict['sp_time'][t0:tf]
        dt = dict['sp_time'][1] - dict['sp_time'][0]
        r = np.shape(data)[0]
        tsize = np.shape(data)[1]
        windowsize = int(np.floor(tsize/float(numwindows)))
        if numwindows==1:
            windowsize=windowsize-1
        if tsize >= windowsize:
            starts = np.linspace(0, \
                int(np.floor(tsize/float(numwindows)))*(numwindows-1),numwindows, dtype='int')
            ends = starts + np.ones(numwindows,dtype='int')*windowsize
        else:
            print('windowsize > tsize, dmd invalid')
        trunc = dict['trunc'] 
        print(starts,ends)
        dmd_b = np.zeros((r,tsize),dtype='complex')
        b_inj = np.zeros((r,tsize),dtype='complex')
        b_eq = np.zeros((r,tsize),dtype='complex')
        b_extra = np.zeros((r,tsize),dtype='complex')
        dict['sdmd_y0'] = []
        dict['sdmd_omega'] = []
        dict['sdmd_phi'] = []
        S = np.zeros((trunc,trunc))
        for gammas in [1.0]:
            for i in range(numwindows):
                X = data[:,starts[i]:ends[i]]
                Xprime = data[:,1+starts[i]:ends[i]+1]
                Udmd,Sdmd,Vdmd = np.linalg.svd(X,full_matrices=False)
                Vdmd = np.transpose(Vdmd)
                Udmd = Udmd[:,0:trunc]
                Sdmd = Sdmd[0:trunc]
                Vdmd = Vdmd[:,0:trunc]
                A = np.dot(np.dot(np.transpose(Udmd),Xprime),Vdmd/Sdmd)
                eigvals,Y = np.linalg.eig(A)
                Phi = np.dot(np.dot(Xprime,Vdmd/Sdmd),Y)
                omega = np.log(eigvals)/dt
                omega[np.isnan(omega).nonzero()] = 0
                decay = abs(np.real(omega)/(pi*2))
                oscillation = abs(np.imag(omega)/(pi*2))
                u_modes = np.zeros((trunc,windowsize),'complex')
                p_modes = np.zeros((trunc,windowsize),'complex')
                Vandermonde = np.zeros((trunc,windowsize),'complex')
                for iter in range(windowsize):
                    u_modes[:,iter] = np.exp((omega-np.real(omega))* \
                        (time[starts[i]+iter]-time[starts[i]]))
                    p_modes[:,iter] = np.exp(np.real(omega)* \
                        (time[starts[i]+iter]-time[starts[i]]))
                    Vandermonde[:,iter] = u_modes[:,iter]*p_modes[:,iter]
                for jj in range(trunc):
                    for kk in range(trunc):
                        if jj == kk:
                            S[jj,kk] = Sdmd[jj]
                y0 = sparse_algorithm(trunc,Y,Vandermonde,S,Vdmd,gammas)
                dict['sdmd_y0'].append(y0)
                dict['sdmd_omega'].append(omega)
                dict['sdmd_phi'] = Phi

                typename = r'sparse DMD, $\gamma$ = '+str(int(gammas))
                filename = 'power_DMD'+str(dictname)+'_'+str(i)+'.png'
                power_spectrum(y0,omega,inj_freq,filename,typename)
                filename = 'phasePlot_DMD'+str(dictname)+'_'+str(i)+'.png'
                freq_phase_plot(y0,omega,inj_freq,filename,typename)

#                equilIndex = np.asarray(np.asarray(abs(np.imag(omega))==0).nonzero())
#                if equilIndex.size==0:
#                    equilIndex = np.atleast_1d(np.argmin(abs(np.imag(omega))))
#                equilIndex = np.ravel(equilIndex).tolist()
                sortd = np.flip(np.argsort(y0*np.conj(y0)))
                equilIndex = np.atleast_1d(sortd[0])
                injIndex = sortd[1:3]
                extraIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
                    abs(np.imag(omega)/(2*pi)),28500,atol=500)).nonzero()))
                sortd = np.flip(np.argsort(y0*np.conj(y0)))
                print(omega[sortd]/(2*pi*1000.0))
                print(y0[sortd]*np.conj(y0[sortd]))
                for iter in range(windowsize):
                    u_modes[:,iter] = y0*np.exp((omega-np.real(omega))* \
                        (time[starts[i]+iter]-time[starts[i]]))
                    p_modes[:,iter] = np.exp(np.real(omega)* \
                        (time[starts[i]+iter]-time[starts[i]]))
                    Vandermonde[:,iter] = u_modes[:,iter]*p_modes[:,iter]
                mag_energy = abs((u_modes*p_modes)*np.conj(u_modes*p_modes) \
                    /np.sum((u_modes*p_modes)*np.conj(u_modes*p_modes)))
                energy_order = np.flip(np.argsort(np.mean(mag_energy,axis=1)))
                dmd_b[:,starts[i]:ends[i]] = 0.5*np.dot(Phi,u_modes*p_modes)
                for j in range(len(equilIndex)):
                    e_ind = equilIndex[j]
                    b_eq[:,starts[i]:ends[i]] += 0.5*np.outer(Phi[:,e_ind], \
                        u_modes[e_ind,:]*p_modes[e_ind,:])
                for j in range(len(injIndex)):
                    i_ind = injIndex[j]
                    b_inj[:,starts[i]:ends[i]] += 0.5*np.outer(Phi[:,i_ind], \
                        u_modes[i_ind,:]*p_modes[i_ind,:])
                for j in range(len(extraIndex)):
                    ex_ind = extraIndex[j]
                    b_extra[:,starts[i]:ends[i]] += 0.5*np.outer(Phi[:,ex_ind], \
                        u_modes[ex_ind,:]*p_modes[ex_ind,:])
                b_inj_phase = np.arctan2( \
                    np.imag(b_inj[:,starts[i]:ends[i]]), \
                    np.real(b_inj[:,starts[i]:ends[i]]))
                b_inj[:,starts[i]:ends[i]] += \
                    np.conj(b_inj[:,starts[i]:ends[i]])
                b_eq_phase = np.arctan2( \
                    np.imag(b_eq[:,starts[i]:ends[i]]), \
                    np.real(b_eq[:,starts[i]:ends[i]]))
                b_eq[:,starts[i]:ends[i]] += \
                    np.conj(b_eq[:,starts[i]:ends[i]])
                b_extra_phase = np.arctan2( \
                    np.imag(b_extra[:,starts[i]:ends[i]]), \
                    np.real(b_extra[:,starts[i]:ends[i]]))
                b_extra[:,starts[i]:ends[i]] += \
                    np.conj(b_extra[:,starts[i]:ends[i]])
                dmd_b[:,starts[i]:ends[i]] += \
                    np.conj(dmd_b[:,starts[i]:ends[i]])
                dmd_b_phase = np.arctan2( \
                    np.imag(dmd_b[:,starts[i]:ends[i]]), \
                    np.real(dmd_b[:,starts[i]:ends[i]]))
                err = np.linalg.norm(X-dmd_b[:,starts[i]:ends[i]],'fro') \
                    /np.linalg.norm(X,'fro')
                print('gamma = ',gammas,', err = ',err)
        dict['sdmd_b'] = dmd_b
        dict['sdmd_b_phase'] = dmd_b_phase
        dict['sdmd_b_eq'] = b_eq
        dict['sdmd_b_eq_phase'] = b_eq_phase
        dict['sdmd_b_inj'] = b_inj
        dict['sdmd_b_inj_phase'] = b_inj_phase
        dict['sdmd_b_extra'] = b_extra
        dict['sdmd_b_extra_phase'] = b_extra_phase

## optimized DMD with variable projection, with sliding window,
## where we minimize |Xt - Vandemonde^T*B|_Frobenius
# @param total A list of psi-tet dictionaries
# @param inj_frequencies Injector frequencies of the dictionaries
# @param numwindows Number of windows for the sliding window
def optimized_DMD_slide(total,inj_frequencies,numwindows):
    fignum = len(total)*numwindows
    for k in range(len(total)):
        dict = total[k]
        inj_freq = inj_frequencies[k]
        dictname = dict['filename']
        t0 = dict['t0']
        tf = dict['tf']
        data = dict['SVD_data']
        time = dict['sp_time'][t0:tf]
        dt = dict['sp_time'][1] - dict['sp_time'][0]
        r = np.shape(data)[0]
        print(r)
        tsize = np.shape(data)[1]
        windowsize = int(np.floor(tsize/float(numwindows)))
        if numwindows==1:
            windowsize=windowsize-1
        if tsize >= windowsize:
            starts = np.linspace(0, \
                int(np.floor(tsize/float(numwindows)))*(numwindows-1),numwindows, dtype='int')
            ends = starts + np.ones(numwindows,dtype='int')*windowsize
        else:
            print('windowsize > tsize, dmd invalid')
        trunc = dict['trunc'] 
        dmd_b = np.zeros((r,tsize),dtype='complex')
        b_inj = np.zeros((r,tsize),dtype='complex')
        b_eq = np.zeros((r,tsize),dtype='complex')
        b_extra = np.zeros((r,tsize),dtype='complex')
        dict['odmd_y0'] = []
        dict['odmd_y0avg'] = []
        dict['odmd_omega'] = []
        dict['odmd_phi'] = []
        #ends[0] = int(tsize*3.5/5.0)
        initialize_variable_project(dict,data,trunc)
        for i in range(numwindows):
            # fit to all of data (algorithm 2 in the paper)
            tbase = time[starts[i]:ends[i]]
            X = data[:,starts[i]:ends[i]]
            tic = Clock.process_time()

            Y,alpha = \
                variable_project(np.transpose(X),dict,trunc,starts[i],ends[i])
            toc = Clock.process_time()
            print('time in variable_projection = ',(toc-tic)*1e-1,' s')
            Y = np.transpose(Y)
            # normalize
            b = np.conj(np.transpose(np.sqrt(np.sum(abs(Y)**2,axis=0))))
            Y = np.dot(Y,np.diag(1.0/b))
            y0 = b
            omega = alpha #np.log(alpha)/dt
            omega[np.isnan(omega).nonzero()] = 0
            decay = abs(np.real(omega)/(pi*2))
            oscillation = abs(np.imag(omega)/(pi*2))
            phimat = variable_project_make_phimat(alpha,tbase-tbase[0])
            dict['odmd_y0'].append(y0)
            dict['odmd_y0avg'].append(y0* \
                np.mean(phimat,axis=0))
            dict['odmd_omega'].append(omega)
            dict['odmd_phi'].append(Y)
            sortd = np.argsort(abs(np.real(omega))/(2*pi*1000.0))
            print(omega[sortd]/(2*pi*1000.0))
            print(y0[sortd]*np.conj(y0[sortd]))
            sortd = np.flip(np.argsort(np.real(omega)/(2*pi*1000.0)))
            equilIndex = np.asarray(np.asarray(abs(np.imag(omega))==0).nonzero())
            if equilIndex.size==0:
                equilIndex = np.atleast_1d(np.argmin(abs(np.imag(omega))))
            equilIndex = np.ravel(equilIndex).tolist()
            injIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
                abs(np.imag(omega)/(2*pi)),inj_freq*1000.0,atol=700)).nonzero()))
            extraIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
                abs(np.imag(omega)/(2*pi)),1000,atol=100)).nonzero()))
            sortd = np.flip(np.argsort(abs(y0)))
            print(omega[sortd]/(2*pi*1000.0))
            print(y0[sortd]*np.conj(y0[sortd]))
            print(extraIndex,injIndex,equilIndex,omega[extraIndex]/(2*pi*1000.0))
            for mode in range(trunc):
                dmd_b[:,starts[i]:ends[i]] += \
                    0.5*y0[mode]*np.outer(Y[:,mode],phimat[:,mode])
                if mode in equilIndex:
                    b_eq[:,starts[i]:ends[i]] += \
                        0.5*y0[mode]*np.outer(Y[:,mode],phimat[:,mode])
                if mode in injIndex:
                    b_inj[:,starts[i]:ends[i]] += \
                        0.5*y0[mode]*np.outer(Y[:,mode],phimat[:,mode])
                if mode in extraIndex:
                    b_extra[:,starts[i]:ends[i]] += \
                        0.5*y0[mode]*np.outer(Y[:,mode],phimat[:,mode])
            typename = 'optimized DMD'
            filename = 'power_DMD'+str(dictname)+'_'+str(i)+'.png'
            power_spectrum(y0,omega,inj_freq,filename,typename)
            filename = 'phasePlot_DMD'+str(dictname)+'_'+str(i)+'.png'
            freq_phase_plot(y0,omega,inj_freq,filename,typename)
            b_inj_phase = np.arctan2( \
                np.imag(b_inj[:,starts[i]:ends[i]]), \
                np.real(b_inj[:,starts[i]:ends[i]]))
            b_inj[:,starts[i]:ends[i]] += \
                np.conj(b_inj[:,starts[i]:ends[i]])
            b_eq_phase = np.arctan2( \
                np.imag(b_eq[:,starts[i]:ends[i]]), \
                np.real(b_eq[:,starts[i]:ends[i]]))
            b_eq[:,starts[i]:ends[i]] += \
                np.conj(b_eq[:,starts[i]:ends[i]])
            b_extra_phase = np.arctan2( \
                np.imag(b_extra[:,starts[i]:ends[i]]), \
                np.real(b_extra[:,starts[i]:ends[i]]))
            b_extra[:,starts[i]:ends[i]] += \
                np.conj(b_extra[:,starts[i]:ends[i]])
            dmd_b_phase = np.arctan2( \
                np.imag(dmd_b[:,starts[i]:ends[i]]), \
                np.real(dmd_b[:,starts[i]:ends[i]]))
            dmd_b[:,starts[i]:ends[i]] += \
                np.conj(dmd_b[:,starts[i]:ends[i]])
        dict['odmd_phi'] = np.reshape(dict['odmd_phi'], \
            (np.shape(X)[0],trunc))
        dict['odmd_b'] = dmd_b
        dict['odmd_b_phase'] = dmd_b_phase
        dict['odmd_b_eq'] = b_eq
        dict['odmd_b_eq_phase'] = b_eq_phase
        dict['odmd_b_inj'] = b_inj
        dict['odmd_b_inj_phase'] = b_inj_phase
        dict['odmd_b_extra'] = b_extra
        dict['odmd_b_extra_phase'] = b_extra_phase

## Performs the sparse DMD algorithm (see the paper)
# @param trunc Truncation number for the SVD
# @param Y Eigenvectors of the matrix A
# @param Vandermonde Matrix of exponentials
# @param S singular value Matrix
# @param Vdmd The V from the SVD of the data matrix X
# @param gamma The sparsity-promotion knob
def sparse_algorithm(trunc,Y,Vandermonde,S,Vdmd,gamma):
    max_iters = 100000
    eps_prime = 1e-9
    eps_dual = 1e-9
    rho = 1.0
    kappa = gamma/rho
    lamda = np.ones((trunc,max_iters),dtype='complex')
    alpha = np.ones((trunc,max_iters),dtype='complex')
    beta = np.zeros((trunc,max_iters),dtype='complex')
    Id = np.identity(trunc)
    q = np.conj(np.diag(np.dot(np.dot(np.dot( \
        Vandermonde,Vdmd),np.conj(S)),Y)))
    P = np.dot(np.conj(np.transpose(Y)),Y)* \
        np.conj(np.dot(Vandermonde, \
        np.conj(np.transpose(Vandermonde))))
    y0_old = np.dot(np.linalg.inv(P),q)
    # Note there is an issue here, that if gamma is very large
    # all the beta[:,j+1] stay zero for all j
    for j in range(max_iters):
        if np.mod(j,1000)==0:
            print(j)
        u = beta[:,j] - lamda[:,j]/rho
        alpha[:,j+1] = np.dot(np.linalg.inv( \
            P+rho/2.0*Id),(q+rho/2.0*u))
        v = alpha[:,j+1] + lamda[:,j]/rho
        ind1 = np.asarray(v > kappa).nonzero()
        ind2 = np.asarray(v < -kappa).nonzero()
        ind3 = np.asarray(abs(v) < kappa).nonzero()
        beta[ind1,j+1] = v[ind1]-kappa
        beta[ind2,j+1] = v[ind2]+kappa
        beta[ind3,j+1] = 0.0
        lamda[:,j+1] = lamda[:,j] + rho*(alpha[:,j+1]-beta[:,j+1])
        if np.dot((alpha[:,j+1]-beta[:,j+1]), \
            np.conj((alpha[:,j+1]-beta[:,j+1]))) < eps_prime \
            and np.dot((beta[:,j+1]-beta[:,j]), \
            np.conj((beta[:,j+1]-beta[:,j]))) < eps_dual:
            max_iters = j
            break
        return alpha[:,j+1]

## Initializes a "good" initial guess for the variable
## projection algorithm used in the optimized DMD
# @param dict A dictionary with initialized SVD data
# @param data The data matrix
# @param trunc Truncation number for the SVD
def initialize_variable_project(dict,data,trunc):
    t0 = dict['t0']
    tf = dict['tf']
    U = dict['U']
    X = data
    r = np.shape(X)[0]
    time = dict['sp_time'][t0:tf]
    tsize = len(time)

    # use projected trapezoidal rule approximation
    # to eigenvalues as initial guess
    ux1 = np.dot(np.transpose(np.conj(U)),X)
    ux2 = ux1[:,2:]
    ux1 = ux1[:,1:np.shape(ux1)[1]-1]

    t1 = time[1:tsize-1]
    t2 = time[2:]

    dx = np.dot((ux2-ux1),np.diag(1.0/(t2-t1)))
    xin = (ux1+ux2)/2

    u1,s1,v1h = np.linalg.svd(xin,full_matrices=False)
    v1 = np.conj(np.transpose(v1h))
    u1 = u1[:,0:trunc]
    v1 = v1[:,0:trunc]
    s1 = np.diag(s1[0:trunc])
    atilde = np.dot(np.dot(np.dot( \
        np.transpose(np.conj(u1)),dx),v1),np.linalg.inv(s1))
    dict['alpha_init'],eigvecs = np.linalg.eig(atilde)

## Performs the Levenberg-Marquadt
## variable projection algorithm for the optimized DMD
## with a parallel qr decomposition using the 
## parallel direct TSQR algorithm (Benson, 2013)
# @param Xt The transposed data matrix (so number of time samples
#   is the number of rows, rather than columns)
# @param dict A dictionary object which has initialized SVD data
#   and initialized first guess for the alphas, defined as alpha_init
# @param trunc The truncation number of the SVD
# @param start The first index to use in the time array
# @param end The last index to use in the time array
# @returns b The coefficients of the optimized DMD reconstruction
# @returns alpha The frequencies in the Vandermonde matrix
# @var lambda0 Initial
#   value used for the regularization parameter
#   lambda in the Levenberg method (a larger
#   lambda makes it more like gradient descent)
# @var maxlam Maximum number
#   of steps used in the inner Levenberg loop,
#   i.e. the number of times you increase lambda
#   before quitting
# @var lamup Factor by which
#   you increase lambda when searching for an
#   appropriate step
# @var lamdown Factor by which
#   you decrease lambda when checking if that
#   results in an error decrease
# @var maxiter The maximum number of outer
#   loop iterations to use before quitting
# @var tol The tolerance for the relative
#   error in the residual, i.e. the program will
#   terminate if algorithm achieves
#   norm(y-Phi(alpha)*b,'fro')/norm(y,'fro') < tol
# @var eps_stall The tolerance for detecting
#   a stall. If err(iter-1)-err(iter) < eps_stall*err(iter-1)
#   then a stall is detected and the program halts.
def variable_project(Xt,dict,trunc,start,end):
    Xt = Xt.astype(np.complex128) #for numba
    lambda0 = 1.0
    maxlam = 20
    lamup = 2.0
    lamdown = lamup
    maxiter = 100
    tol = 1e-3
    eps_stall = 1e-4
    m = np.shape(Xt)[0]
    r = np.shape(Xt)[1]
    n = r
    t0 = dict['t0']
    tf = dict['tf']
    time = dict['sp_time'][t0:tf]
    time = time[start:end]-time[0]
    tsize = len(time)
    dt = time[1]-time[0]
    # initialize values
    #alpha = np.ravel(dict['dmd_omega'])
    alpha = dict['alpha_init']
    osort = np.argsort(np.real(alpha))
    print(alpha[osort]/(2*pi*1000.0))
    alphas = np.zeros((trunc,maxiter),dtype='complex')
    err = np.zeros(maxiter)
    res_scale = np.linalg.norm(Xt,'fro')
    scales = np.zeros(trunc)
    djacmat = np.zeros((m*r,trunc),dtype='complex')
    phimat = variable_project_make_phimat(alpha,time);
    U,S,Vh = np.linalg.svd(phimat,full_matrices=False)
    V = np.conj(np.transpose(Vh))
    U = U[:,0:trunc]
    S = np.diag(S[0:trunc])
    V = V[:,0:trunc]
    Sinv = np.linalg.inv(S).astype(np.complex128)
    b = parallel_lstsq(phimat,Xt)
    #b,g1,g2,g3 = np.linalg.lstsq(phimat,Xt)
    res = Xt - np.dot(phimat,b)
    errlast = np.linalg.norm(res,'fro')/res_scale
    imode = 0
    numThreads = dict['nprocs']
    config.NUMBA_NUM_THREADS=numThreads
    #config.NUMBA_ENABLE_PROFILING=True
    #config.DEBUG=True
    print('NUMBA, using ',config.NUMBA_NUM_THREADS,' threads')
    Rprime = np.zeros((trunc*numThreads,trunc),complex)
    Q1 = np.zeros((m*r,trunc),dtype=complex)
    #Q2 = np.zeros((trunc*numThreads,trunc),dtype=np.complex128)
    Q = np.zeros((m*r,trunc),dtype=complex)
    #dphitemp = np.zeros((tsize,len(alpha)),dtype=np.complex128)
    skip = int(m*r/numThreads)
    for iter in range(maxiter):
        print('iter=',iter,', err=',errlast)
        tic_total = Clock.time()
        djacmat,scales = make_jacobian(alpha,time,U,Sinv,Vh, \
            res,b,trunc,djacmat,scales)
        Q1,Rprime = TSQR1(djacmat,trunc,numThreads,Q1,Rprime,skip)
        Q2,R = TSQR2(Rprime)
        #Q2,R = np.linalg.qr(Rprime)
        Q = TSQR3(Q1,Q2,trunc,numThreads,Q,skip)
        rjac = R
        rhstop = np.dot(np.conj(np.transpose(Q)), \
            np.ravel(res,order='F'))
        toc_total = Clock.time()
        print('QR time = ',toc_total-tic_total)
        rhstop = np.reshape(rhstop,(len(rhstop),1))
        rhs = np.ravel(np.vstack((rhstop, np.zeros((trunc,1))))) # transformed right hand side

        A = np.vstack((rjac,lambda0*np.diag(scales)))
        delta0 = parallel_lstsq(A,rhs)
        #delta0,g1,g2,g3 = np.linalg.lstsq(A,rhs)
        # new alpha guess
        alpha0 = alpha - np.ravel(delta0)
        # corresponding residual
        phimat = variable_project_make_phimat(alpha0,time)

        b0 = parallel_lstsq(phimat,Xt)
        #b0,g1,g2,g3 = np.linalg.lstsq(phimat,Xt)
        res0 = Xt-np.dot(phimat,b0)
        err0 = np.linalg.norm(res0,'fro')/res_scale
        # check if this is an improvement

        if (err0 < errlast):

            # see if a smaller lambda is better
            lambda1 = lambda0/lamdown
            A = np.vstack((rjac,lambda1*np.diag(scales)))
            delta1 = parallel_lstsq(A,rhs)
            #delta1,g1,g2,g3 = np.linalg.lstsq(A,rhs)
            alpha1 = alpha - np.ravel(delta1)
            phimat = variable_project_make_phimat(alpha1,time)
            b1 = parallel_lstsq(phimat,Xt)
            #b1,g1,g2,g3 = np.linalg.lstsq(phimat,Xt)
            res1 = Xt-np.dot(phimat,b1)
            err1 = np.linalg.norm(res1,'fro')/res_scale

            if (err1 < err0):
                lambda0 = lambda1
                alpha = alpha1
                errlast = err1
                b = b1
                res = res1
            else:
                alpha = alpha0
                errlast = err0
                b = b0
                res = res0
        else:
        # if not, increase lambda until something works
        # this makes the algorithm more and more like gradient descent
            for j in range(maxlam):
                print(lambda0)
                lambda0 = lambda0*lamup
                A = np.vstack((rjac,lambda0*np.diag(scales)))
                delta0 = parallel_lstsq(A,rhs)
                #delta0,g1,g2,g3 = np.linalg.lstsq(A,rhs)
                alpha0 = alpha - np.ravel(delta0)
                phimat = variable_project_make_phimat(alpha0,time)
                b0 = parallel_lstsq(phimat,Xt)
                #b0,g1,g2,g3 = np.linalg.lstsq(phimat,Xt)
                res0 = Xt-np.dot(phimat,b0)
                err0 = np.linalg.norm(res0,'fro')/res_scale
                if (err0 < errlast):
                    break

            if (err0 < errlast):
                alpha = alpha0
                errlast = err0
                b = b0
                res = res0
            else:
                # no appropriate step length found
                print('Failed to find appropriate step length:')
                print(' iter=',iter,', err=',errlast)
                return b,alpha

        alphas[:,iter] = alpha
        err[iter] = errlast

        if (errlast < tol):
            niter = iter
            return b,alpha

        # stall detection
        if (iter > 1):
            if (err[iter-1]-err[iter] < eps_stall*err[iter-1]):
                niter = iter
                print('algorithm stalled')
                return b,alpha

        phimat = variable_project_make_phimat(alpha,time)
        U,S,Vh = np.linalg.svd(phimat,full_matrices=False)
        V = np.conj(np.transpose(Vh))
        U = U[:,0:trunc]
        S = np.diag(S[0:trunc])
        V = V[:,0:trunc]

    niter = maxiter
    print('Failed to meet tolerance after maxiter steps')
    return b,alpha

## Make the Vandermonde matrix using numba
# @param alpha The frequency part in e^(alpha*time)
# @param time The time base 
# @returns VandermondeT Vandermonde^T
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def variable_project_make_phimat(alpha,time):
    VandermondeT = np.exp(np.outer(time,alpha))
    return VandermondeT 

## A function to make the derivative of the Vandermonde
## with respect to alpha_i
# @param alpha The frequency part in e^(alpha*time)
# @param time The time base 
# @param i The alpha_i index for the derivative
# @returns A Matrix derivative(Vandermonde^T) with respect to alpha_i
def variable_project_make_dphimat(alpha,time,i):
    m = len(time)
    n = len(alpha)
    A = np.zeros((m,n),dtype='complex')
    A[:,i] = time*np.exp(alpha[i]*time)
    return A

## Creates the Jacobian for the Levenberg-Marquadt solution
## of the optimized DMD algorithm. This is parallelized with numba
# @param alpha The frequency part in e^(alpha*time)
# @param time The time base 
# @param U The U from the SVD of the Vandermonde
# @param Sinv This is inv(S) where S is from the SVD of the Vandermonde
# @param Vh This is V* from the SVD of the Vandermonde
# @param res The residual (error) = |Xt-Vandermonde^T*B|
# @param b The optimized DMD coefficients b_k
# @param trunc The truncation number for the SVD
# @param djacmat The Jacobian from the previous iteration, 
#   this is overwritten
# @param scales The scales from the previous iteration, 
#   this is overwritten 
# @returns djacmat The Jacobian needed for Levenberg-Marquadt
# @returns scales Forms the diagonal of the 'M' matrix for 'Marquadt'
#   part of the algorithm
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def make_jacobian(alpha,time,U,Sinv,Vh,res,b,trunc,djacmat,scales):
    mh = len(time)
    nh = len(alpha)
    for j in range(trunc):
        dphitemp = np.zeros((mh,nh),np.complex128)
        dphitemp[:,j] = time*np.exp(alpha[j]*time)
        djaca = np.dot(dphitemp - \
            np.dot(U,np.dot(np.transpose(U.conj()), \
            dphitemp)),b)
        djacb = np.dot(U,np.dot(Sinv,
            np.dot(Vh, \
            np.dot(np.transpose(dphitemp.conj()),res))))
        djacmat[:,j] = -np.ravel(np.transpose(djaca))-np.ravel(np.transpose(djacb))
        # the scales give the "marquardt" part of the algo.
        scales[j] = min(np.linalg.norm(djacmat),1)
        scales[j] = max(scales[j],1e-6)
    return djacmat,scales

## Performs parallel qr decompositions, using numba,
## to construct the Q1 matrix  
# @param djacmat The Jacobian needed for Levenberg-Marquadt
# @param trunc The truncation number for the SVD, and the 
#   row size of each of the R submatrices resulting from each
#   of the qr decompositions
# @param numThreads The number of threads being used by numba
# @param Q1 The Q1 from the previous iteration, this is overwritten
# @param Rprime The Rprime from the previous iteration, this is overwritten
# @param skip The row size of each of the Q matrices resulting
#   from each of the qr decompositions
# @returns Q1 The full matrix resulting from concatenation of the 
#   Q from the separate qr decompositions
# @returns Rprime The full matrix resulting from concatenation of
#   the R from the separate qr decompositions
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def TSQR1(djacmat,trunc,numThreads,Q1,Rprime,skip):
    for x in range(numThreads):
       Q1[x*skip:(x+1)*skip,:], \
           Rprime[x*trunc:(x+1)*trunc,:] = \
           np.linalg.qr( \
           djacmat[x*skip:(x+1)*skip,:])
    return Q1,Rprime


## Takes the qr decomposition of Rprime
# @param Rprime the matrix resulting from concatenation of the 
# R from the separate qr decompositions
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def TSQR2(Rprime):
    return np.linalg.qr(Rprime)

## Performs parallel matrix products, using numba,
## to construct our final Q = Q1*Q2
# @param Q1 The full matrix resulting from concatenation of the 
#   Q from the separate qr decompositions
# @param Q2 The Q from the qr decomposition of Rprime 
# @param trunc The truncation number for the SVD, and the 
#   row size of each of the Q2 submatrices  
# @param numThreads The number of threads being used by numba
# @param Q The Q from the previous iteration, this is overwritten
# @param skip The row size of each of the Q and Q1 submatrices
# @returns Q The final Q we want from the parallelized qr 
#   decomposition of the Jacobian
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def TSQR3(Q1,Q2,trunc,numThreads,Q,skip):
    for x in range(numThreads):
        Q[x*skip:(x+1)*skip,:] = \
            np.dot( \
            Q1[x*skip:(x+1)*skip,:], \
            Q2[x*trunc:(x+1)*trunc,:])
    return Q

## Performs parallel lstsq on Ax=b using numba
# @param A The A matrix in Ax = b
# @param b The b matrix in Ax = b
# @returns x The x matrix in Ax = b
@jit(nopython=True,parallel=True,nogil=True,cache=True)
def parallel_lstsq(A,b):
    x,g1,g2,g3 = np.linalg.lstsq(A,b)
    return x
