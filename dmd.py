## @package dmd
## This file contains the different dmd methods
from plot_attributes import *
from dmd_utilities import power_spectrum, freq_phase_plot
from numba import jit,config
import time as Clock

## DMD routine, with sliding window,
## where we minimize |Xt - Vandermonde^T*B|_Frobenius
# @param total A list of psi-tet dictionaries
# @param numwindows Number of windows for the sliding window
# @param dmd_flag Flag to indicate which DMD method to use
def DMD_slide(total,numwindows,dmd_flag):
    fignum = len(total)*numwindows
    for k in range(len(total)):
        dict = total[k]
        f_1 = dict['f_1']
        dictname = dict['filename']
        t0 = dict['t0']
        tf = dict['tf']
        data = np.copy(dict['SVD_data'])
        time = dict['sp_time'][t0:tf-1]
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
        Bfield = np.zeros((r,tsize),dtype='complex')
        Bfield_inj = np.zeros((r,tsize),dtype='complex')
        Bfield_eq = np.zeros((r,tsize),dtype='complex')
        Bfield_anom = np.zeros((r,tsize),dtype='complex')
        dmd_b = []
        dmd_omega = []
        for i in range(numwindows):
            tbase = time[starts[i]:ends[i]]
            X = data[:,starts[i]:ends[i]]
            if dmd_flag == 1 or dmd_flag == 2:
                Xprime = data[:,1+starts[i]:ends[i]+1]
                Udmd,Sdmd,Vdmd = np.linalg.svd(X,full_matrices=False)
                Vdmd = np.transpose(Vdmd)
                Udmd = Udmd[:,0:trunc]
                Sdmd = Sdmd[0:trunc]
                Vdmd = Vdmd[:,0:trunc]
                S = np.diag(Sdmd)
                A = np.dot(np.dot(np.transpose(Udmd),Xprime),Vdmd/Sdmd)
                eigvals,Y = np.linalg.eig(A)
                Bt = np.dot(np.dot(Xprime,Vdmd/Sdmd),Y)
                omega = np.log(eigvals)/dt
                VandermondeT = make_VandermondeT(omega,tbase-tbase[0])
                Vandermonde = np.transpose(VandermondeT)
                q = np.conj(np.diag(np.dot(np.dot(np.dot( \
                    Vandermonde,Vdmd),np.conj(S)),Y)))
                P = np.dot(np.conj(np.transpose(Y)),Y)* \
                    np.conj(np.dot(Vandermonde, \
                    np.conj(VandermondeT)))
                b = np.dot(np.linalg.inv(P),q)
                typename = 'DMD'
                if dmd_flag == 2:
                    gamma = 10.0
                    b = sparse_algorithm(trunc,q,P,b,gamma)
                    typename = 'sparse DMD'
            elif dmd_flag == 3:
                initialize_variable_project(dict,data,trunc)
                # Time algorithm 2 from Askham/Kutz 2017
                tic = Clock.time()
                B,omega = variable_project( \
                    np.transpose(X),dict,trunc,starts[i],ends[i])
                toc = Clock.time()
                print('time in variable_projection = ',toc-tic,' s')
                Bt = np.transpose(B)
                b = np.conj(np.transpose(np.sqrt(np.sum(abs(Bt)**2,axis=0))))
                Bt = np.dot(Bt,np.diag(1.0/b))
                typename = 'optimized DMD'
                VandermondeT = make_VandermondeT(omega,tbase-tbase[0])
                Vandermonde = np.transpose(VandermondeT)

            omega[np.isnan(omega).nonzero()] = 0
            dmd_b.append(b)
            dmd_omega.append(omega)
            dmd_Bt = Bt
            #sortd = np.argsort(abs(np.real(omega))/(2*pi*1000.0))
            sortd = np.flip(np.argsort(np.real(omega)/(2*pi*1000.0)))
            print(omega[sortd]/(2*pi*1000.0))
            print(b[sortd]*np.conj(b[sortd]))
            anomIndex = np.atleast_1d(sortd[0:20])
            equilIndex = np.asarray(np.asarray(abs(np.imag(omega))==0).nonzero())
            if equilIndex.size==0:
                equilIndex = np.atleast_1d(np.argmin(abs(np.imag(omega))))
            equilIndex = np.ravel(equilIndex).tolist()
            injIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
                abs(np.imag(omega)/(2*pi)),f_1*1000.0,atol=700)).nonzero()))
            #anomIndex = np.ravel(np.asarray(np.asarray(np.isclose( \
            #    abs(np.imag(omega)/(2*pi)),14500,atol=1000)).nonzero()))
            sortd = np.flip(np.argsort(abs(b)))
            print(omega[sortd]/(2*pi*1000.0))
            print(b[sortd]*np.conj(b[sortd]))
            print(anomIndex,injIndex,equilIndex,omega[anomIndex]/(2*pi*1000.0))
            for mode in range(trunc):
                Bfield[:,starts[i]:ends[i]] += \
                    0.5*b[mode]*np.outer(Bt[:,mode],Vandermonde[mode,:])
                if mode in equilIndex:
                    Bfield_eq[:,starts[i]:ends[i]] += \
                        0.5*b[mode]*np.outer(Bt[:,mode],Vandermonde[mode,:])
                if mode in injIndex:
                    Bfield_inj[:,starts[i]:ends[i]] += \
                        0.5*b[mode]*np.outer(Bt[:,mode],Vandermonde[mode,:])
                if mode in anomIndex:
                    Bfield_anom[:,starts[i]:ends[i]] += \
                        0.5*b[mode]*np.outer(Bt[:,mode],Vandermonde[mode,:])
            Bfield_inj[:,starts[i]:ends[i]] += \
                np.conj(Bfield_inj[:,starts[i]:ends[i]])
            Bfield_eq[:,starts[i]:ends[i]] += \
                np.conj(Bfield_eq[:,starts[i]:ends[i]])
            Bfield_anom[:,starts[i]:ends[i]] += \
                np.conj(Bfield_anom[:,starts[i]:ends[i]])
            Bfield[:,starts[i]:ends[i]] += \
                np.conj(Bfield[:,starts[i]:ends[i]])
            err = np.linalg.norm(X-Bfield[:,starts[i]:ends[i]],'fro') \
                /np.linalg.norm(X,'fro')
            print('Final error = ',err)
            filename = 'power_'+str(dictname[:len(dictname)-4])+'_'+str(i)+'.png'
            power_spectrum(b,omega,f_1,filename,typename)
            filename = 'phasePlot_'+str(dictname[:len(dictname)-4])+'_'+str(i)+'.png'
            freq_phase_plot(b,omega,f_1,filename,typename)

        if dmd_flag == 1:
            dict['Bfield'] = Bfield
            dict['Bfield_eq'] = Bfield_eq
            dict['Bfield_inj'] = Bfield_inj
            dict['Bfield_anom'] = Bfield_anom
            dict['b'] = np.asarray(dmd_b)
            dict['omega'] = np.asarray(dmd_omega)
            dict['Bt'] = dmd_Bt
        elif dmd_flag == 2:
            dict['sparse_Bfield'] = Bfield
            dict['sparse_Bfield_eq'] = Bfield_eq
            dict['sparse_Bfield_inj'] = Bfield_inj
            dict['sparse_Bfield_anom'] = Bfield_anom
            dict['sparse_b'] = np.asarray(dmd_b)
            dict['sparse_omega'] = np.asarray(dmd_omega)
            dict['sparse_Bt'] = dmd_Bt
        elif dmd_flag == 3:
            dict['optimized_Bfield'] = Bfield
            dict['optimized_Bfield_eq'] = Bfield_eq
            dict['optimized_Bfield_inj'] = Bfield_inj
            dict['optimized_Bfield_anom'] = Bfield_anom
            dict['optimized_b'] = np.asarray(dmd_b)
            dict['optimized_omega'] = np.asarray(dmd_omega)
            dict['optimized_Bt'] = dmd_Bt

## Tests the DMD methods on forecasting by
## dividing into test/train data and using
## the full DMD reconstructions
# @param dict A psi-tet dictionary
def DMD_forecast(dict):
    f_1 = dict['f_1']
    dictname = dict['filename']
    t0 = dict['t0']
    tf = dict['tf']
    data = np.copy(dict['SVD_data'])
    time = dict['sp_time'][t0:tf]
    dt = dict['sp_time'][1] - dict['sp_time'][0]
    inj_curr_end = 2
    if dict['is_HITSI3'] == True:
        inj_curr_end = 3
    plt.figure(30000,figsize=(figx, figy))
    plt.grid(True)
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    index = size_bpol
    for i in range(1,4):
        plt.subplot(3,1,i)
        plt.plot(time*1000, \
            dict['SVD_data'][index+inj_curr_end,:]*1e4,'k',
            linewidth=lw)
    r = np.shape(data)[0]
    tsize = np.shape(data)[1]
    trainsize = int(tsize*3.0/5.0)
    testsize = tsize-trainsize
    trunc = dict['trunc']
    dmd_data = np.zeros((r,tsize),dtype='complex')
    sdmd_data = np.zeros((r,tsize),dtype='complex')
    odmd_data = np.zeros((r,tsize),dtype='complex')
    dmd_data[:,0:trainsize] = np.copy(data[:,0:trainsize])
    sdmd_data[:,0:trainsize] = np.copy(data[:,0:trainsize])
    odmd_data[:,0:trainsize] = np.copy(data[:,0:trainsize])
    b = np.ravel(dict['b'])
    omega = np.ravel(dict['omega'])
    Bt = dict['Bt']
    sparse_b = np.ravel(dict['sparse_b'])
    sparse_omega = np.ravel(dict['sparse_omega'])
    sparse_Bt = dict['sparse_Bt']
    optimized_b = np.ravel(dict['optimized_b'])
    optimized_omega = np.ravel(dict['optimized_omega'])
    optimized_Bt = dict['optimized_Bt']
    Vandermonde = \
        np.transpose(make_VandermondeT(omega,time-time[0]))
    sparse_Vandermonde = \
        np.transpose(make_VandermondeT(sparse_omega,time-time[0]))
    optimized_Vandermonde = \
        np.transpose(make_VandermondeT(optimized_omega,time-time[0]))
    for mode in range(trunc):
        dmd_data[:,trainsize+1:] += \
            0.5*b[mode]*np.outer(Bt[:,mode], \
            Vandermonde[mode,trainsize+1:])
        sdmd_data[:,trainsize+1:] += \
            0.5*b[mode]*np.outer(sparse_Bt[:,mode], \
            sparse_Vandermonde[mode,trainsize+1:])
        odmd_data[:,trainsize+1:] += \
            0.5*b[mode]*np.outer(optimized_Bt[:,mode], \
            optimized_Vandermonde[mode,trainsize+1:])
    dmd_data[:,trainsize+1:] += np.conj(dmd_data[:,trainsize+1:])
    sdmd_data[:,trainsize+1:] += np.conj(sdmd_data[:,trainsize+1:])
    odmd_data[:,trainsize+1:] += np.conj(odmd_data[:,trainsize+1:])
    err1 = np.linalg.norm(data[:,trainsize+1:] \
        -dmd_data[:,trainsize+1:],'fro') \
        /np.linalg.norm(data[:,trainsize+1:],'fro')
    err2 = np.linalg.norm(data[:,trainsize+1:] \
        -sdmd_data[:,trainsize+1:],'fro') \
        /np.linalg.norm(data[:,trainsize+1:],'fro')
    err3 = np.linalg.norm(data[:,trainsize+1:] \
        -odmd_data[:,trainsize+1:],'fro') \
        /np.linalg.norm(data[:,trainsize+1:],'fro')
    print('dmderr,sdmd_err,odmderr=',err1,' ',err2,' ',err3)
    plt.subplot(3,1,1)
    plt.plot(time*1000, \
        dmd_data[index+inj_curr_end,:]*1e4,'b',label='DMD Forecast', \
        linewidth=lw, \
        path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
        pe.Normal()])

    plt.subplot(3,1,3)
    plt.plot(time*1000, \
        sdmd_data[index+inj_curr_end,:]*1e4,'r',label='sparse DMD Forecast', \
        linewidth=lw, \
        path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
        pe.Normal()])
    plt.subplot(3,1,2)
    plt.plot(time*1000, \
        odmd_data[index+inj_curr_end,:]*1e4,'g',label='optimized DMD Forecast', \
        linewidth=lw, \
        path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
        pe.Normal()])
    for i in range(1,4):
        plt.subplot(3,1,i)
        if i==1:
            plt.title(dict['filename'][7:13]+', Probe: B_L01T000', \
                fontsize=fs)
        if i==3:
            plt.xlabel('Time (ms)',fontsize=fs)
        plt.ylabel('B (G)',fontsize=fs)
        plt.axvline(x=time[trainsize]*1000,color='k', \
            linewidth=lw)
        plt.legend(loc='upper left',fontsize=ls)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.ylim(-150,300)
        #plt.ylim(-500,1000)
        #ax.set_yticks([-500,0,500,1000])
        ax.set_yticks([-150,0,150,300])
    plt.savefig(out_dir+'forecasting.png')

## Performs the sparse DMD algorithm (see Jovanovic 2014)
# @param trunc Truncation number for the SVD
# @param q Defined in the sparse DMD paper
# @param P Defined in the sparse DMD paper
# @param b The DMD coefficients to be altered
# @param gamma The sparsity-promotion knob
def sparse_algorithm(trunc,q,P,b,gamma):
    max_iters = 100000
    eps_prime = 1e-9
    eps_dual = 1e-9
    rho = 1.0
    kappa = gamma/rho
    lamda = np.ones((trunc,max_iters),dtype='complex')
    alpha = np.ones((trunc,max_iters),dtype='complex')
    beta = np.zeros((trunc,max_iters),dtype='complex')
    Id = np.identity(trunc)
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
    dict['omega_init'],eigvecs = np.linalg.eig(atilde)

## Performs the Levenberg-Marquadt
## variable projection algorithm for the optimized DMD
## with a parallel qr decomposition using the
## parallel direct TSQR algorithm (Benson, 2013)
# @param Xt The transposed data matrix (so number of time samples
#   is the number of rows, rather than columns)
# @param dict A dictionary object which has initialized SVD data
#   and initialized first guess for the omegas, defined as omega_init
# @param trunc The truncation number of the SVD
# @param start The first index to use in the time array
# @param end The last index to use in the time array
# @returns b The coefficients of the optimized DMD reconstruction
# @returns omega The frequencies in the Vandermonde matrix
def variable_project(Xt,dict,trunc,start,end):
    Xt = Xt.astype(np.complex128) #for numba
    ## Initial
    ##   value used for the regularization parameter
    ##   lambda in the Levenberg method (a larger
    ##   lambda makes it more like gradient descent)
    lambda0 = 1.0
    ## Maximum number
    ##   of steps used in the inner Levenberg loop,
    ##   i.e. the number of times you increase lambda
    ##   before quitting
    maxlam = 20
    ## Factor by which
    ##   you increase lambda when searching for an
    ##   appropriate step
    lamup = 2.0
    ## Factor by which
    ##   you decrease lambda when checking if that
    ##   results in an error decrease
    lamdown = lamup
    ## The maximum number of outer
    ##   loop iterations to use before quitting
    maxiter = 100
    ## The tolerance for the relative
    ##   error in the residual, i.e. the program will
    ##   terminate if algorithm achieves err < tol
    tol = 1e-3
    ## The tolerance for detecting
    ##   a stall. If err(iter-1)-err(iter) < eps_stall*err(iter-1)
    ##   then a stall is detected and the program halts.
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
    #omega = np.ravel(dict['dmd_omega'])
    omega = dict['omega_init']
    osort = np.argsort(np.real(omega))
    print(omega[osort]/(2*pi*1000.0))
    omegas = np.zeros((trunc,maxiter),dtype='complex')
    err = np.zeros(maxiter)
    res_scale = np.linalg.norm(Xt,'fro')
    scales = np.zeros(trunc)
    djacmat = np.zeros((m*r,trunc),dtype='complex')
    VandermondeT = make_VandermondeT(omega,time);
    U,S,Vh = np.linalg.svd(VandermondeT,full_matrices=False)
    V = np.conj(np.transpose(Vh))
    U = U[:,0:trunc]
    S = np.diag(S[0:trunc])
    V = V[:,0:trunc]
    Sinv = np.linalg.inv(S).astype(np.complex128)
    b = parallel_lstsq(VandermondeT,Xt)
    res = Xt - np.dot(VandermondeT,b)
    errlast = np.linalg.norm(res,'fro')/res_scale
    imode = 0
    numThreads = dict['nprocs']
    config.NUMBA_NUM_THREADS=numThreads
    print('NUMBA, using ',config.NUMBA_NUM_THREADS,' threads')
    Rprime = np.zeros((trunc*numThreads,trunc),complex)
    Q1 = np.zeros((m*r,trunc),dtype=complex)
    Q = np.zeros((m*r,trunc),dtype=complex)
    skip = int(m*r/numThreads)
    for iter in range(maxiter):
        print('iter=',iter,', err=',errlast)
        tic_total = Clock.time()
        djacmat,scales = make_jacobian(omega,time,U,Sinv,Vh, \
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
        # new omega guess
        omega0 = omega - np.ravel(delta0)
        # corresponding residual
        VandermondeT = make_VandermondeT(omega0,time)

        b0 = parallel_lstsq(VandermondeT,Xt)
        #b0,g1,g2,g3 = np.linalg.lstsq(VandermondeT,Xt)
        res0 = Xt-np.dot(VandermondeT,b0)
        err0 = np.linalg.norm(res0,'fro')/res_scale
        # check if this is an improvement

        if (err0 < errlast):

            # see if a smaller lambda is better
            lambda1 = lambda0/lamdown
            A = np.vstack((rjac,lambda1*np.diag(scales)))
            delta1 = parallel_lstsq(A,rhs)
            #delta1,g1,g2,g3 = np.linalg.lstsq(A,rhs)
            omega1 = omega - np.ravel(delta1)
            VandermondeT = make_VandermondeT(omega1,time)
            b1 = parallel_lstsq(VandermondeT,Xt)
            #b1,g1,g2,g3 = np.linalg.lstsq(VandermondeT,Xt)
            res1 = Xt-np.dot(VandermondeT,b1)
            err1 = np.linalg.norm(res1,'fro')/res_scale

            if (err1 < err0):
                lambda0 = lambda1
                omega = omega1
                errlast = err1
                b = b1
                res = res1
            else:
                omega = omega0
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
                omega0 = omega - np.ravel(delta0)
                VandermondeT = make_VandermondeT(omega0,time)
                b0 = parallel_lstsq(VandermondeT,Xt)
                #b0,g1,g2,g3 = np.linalg.lstsq(VandermondeT,Xt)
                res0 = Xt-np.dot(VandermondeT,b0)
                err0 = np.linalg.norm(res0,'fro')/res_scale
                if (err0 < errlast):
                    break

            if (err0 < errlast):
                omega = omega0
                errlast = err0
                b = b0
                res = res0
            else:
                # no appropriate step length found
                print('Failed to find appropriate step length:')
                print(' iter=',iter,', err=',errlast)
                return b,omega

        omegas[:,iter] = omega
        err[iter] = errlast

        if (errlast < tol):
            niter = iter
            return b,omega

        # stall detection
        if (iter > 1):
            if (err[iter-1]-err[iter] < eps_stall*err[iter-1]):
                niter = iter
                print('algorithm stalled')
                return b,omega

        VandermondeT = make_VandermondeT(omega,time)
        U,S,Vh = np.linalg.svd(VandermondeT,full_matrices=False)
        V = np.conj(np.transpose(Vh))
        U = U[:,0:trunc]
        S = np.diag(S[0:trunc])
        V = V[:,0:trunc]

    niter = maxiter
    print('Failed to meet tolerance after maxiter steps')
    return b,omega

@jit(nopython=True,parallel=True,nogil=True,cache=True)
## Make the Vandermonde matrix using numba
# @param omega The frequency part in e^(omega*time)
# @param time The time base
# @returns VandermondeT Vandermonde^T
def make_VandermondeT(omega,time):
    VandermondeT = np.exp(np.outer(time,omega))
    return VandermondeT

@jit(nopython=True,parallel=True,nogil=True,cache=True)
## Creates the Jacobian for the Levenberg-Marquadt solution
## of the optimized DMD algorithm. This is parallelized with numba
# @param omega The frequency part in e^(omega*time)
# @param time The time base
# @param U The U from the SVD of the Vandermonde
# @param Sinv This is inv(S) where S is from the SVD of the Vandermonde
# @param Vh This is V* from the SVD of the Vandermonde
# @param res The residual (error) = |Xt-Vandermonde^T*B|
# @param b The DMD coefficients
# @param trunc The truncation number for the SVD
# @param djacmat The Jacobian from the previous iteration,
#   this is overwritten
# @param scales The scales from the previous iteration,
#   this is overwritten
# @returns djacmat The Jacobian needed for Levenberg-Marquadt
# @returns scales Forms the diagonal of the 'M' matrix for 'Marquadt'
#   part of the algorithm
def make_jacobian(omega,time,U,Sinv,Vh,res,b,trunc,djacmat,scales):
    mh = len(time)
    nh = len(omega)
    for j in range(trunc):
        dphitemp = np.zeros((mh,nh),np.complex128)
        dphitemp[:,j] = time*np.exp(omega[j]*time)
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

@jit(nopython=True,parallel=True,nogil=True,cache=True)
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
def TSQR1(djacmat,trunc,numThreads,Q1,Rprime,skip):
    for x in range(numThreads):
       Q1[x*skip:(x+1)*skip,:], \
           Rprime[x*trunc:(x+1)*trunc,:] = \
           np.linalg.qr( \
           djacmat[x*skip:(x+1)*skip,:])
    return Q1,Rprime

@jit(nopython=True,parallel=True,nogil=True,cache=True)
## Takes the qr decomposition of Rprime
# @param Rprime the matrix resulting from concatenation of the
# R from the separate qr decompositions
def TSQR2(Rprime):
    return np.linalg.qr(Rprime)

@jit(nopython=True,parallel=True,nogil=True,cache=True)
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
def TSQR3(Q1,Q2,trunc,numThreads,Q,skip):
    for x in range(numThreads):
        Q[x*skip:(x+1)*skip,:] = \
            np.dot( \
            Q1[x*skip:(x+1)*skip,:], \
            Q2[x*trunc:(x+1)*trunc,:])
    return Q

@jit(nopython=True,parallel=True,nogil=True,cache=True)
## Performs parallel lstsq on Ax=b using numba
# @param A The A matrix in Ax = b
# @param b The b matrix in Ax = b
# @returns x The x matrix in Ax = b
def parallel_lstsq(A,b):
    x,g1,g2,g3 = np.linalg.lstsq(A,b)
    return x
