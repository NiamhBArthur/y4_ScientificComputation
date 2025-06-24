"""Scientific Computation Project 3
01864994
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.sparse
import scipy.linalg
from math import *

#===== Code for Part 1=====#

def plot_field(lat,lon,u,time,levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()
    
    return None

def part1():#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load('data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #-------------------------------------#

    # visualisations
    plot_field(lat, lon, u, 0)
    plot_field(lat, lon, u, 1)
    plot_field(lat, lon, u, 2)
    
    # reshape and perform PCA to centered data
    X = u.reshape((u.shape[0], u.shape[1] * u.shape[2]))
    M, N = X.shape
    A_spatial= np.transpose(X-X.mean(axis=0))
    A_temporal = np.transpose(X.T-X.mean(axis=1))
    U_spatial, _, _ = np.linalg.svd(A_spatial)
    U_temporal, _, _ = np.linalg.svd(A_temporal)

    # find the variance explained at each principal component
    Cov = 1/(N-1)*A_spatial.dot(A_spatial.T)
    total_variance = np.sum(np.diag(Cov))
    evals, evects = np.linalg.eig(Cov)
    plt.plot((np.cumsum(np.abs(evals))/total_variance)[:100], label='Spatial Principal Components', color='blue')
    Covq = 1/(M-1)*A_temporal.dot(A_temporal.T)
    total_variance = np.sum(np.diag(Covq))
    evals, evects = np.linalg.eig(Covq)
    plt.plot((np.cumsum(np.abs(evals))/total_variance)[:100], label='Temporal Principal Components', color='orange')
    plt.title('Cumulative Variance explained by the first 100 Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion Variance')
    plt.legend()
    plt.show()

    for i in range(2): # make plots for the first two principal components
        plt.plot(U_spatial[:, i], color='blue')
        plt.title(f'{i}th Spatial Principal component')
        plt.xlabel('i')
        plt.ylabel('ith value in PC')
        plt.show()


        plt.contourf(lon, lat, (U_spatial[:, i]).reshape((16,144)))
        plt.title(f'{i}th Spatial Principal component, reshaped')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

        plt.plot(A_spatial.T @ U_spatial[:, i], color='blue')
        plt.xlabel('time')
        plt.ylabel('x̃(t)')
        plt.title(f'$x̃_{i}$')
        plt.show()

        # perform fft, shift to be in order for the plot. take absolute value to find magnitude of peaks
        x = np.fft.fftshift(np.fft.fft(A_spatial.T @ U_spatial[:, i] - np.mean(A_spatial.T @ U_spatial[:, i])))
        plt.plot(np.fft.fftshift(np.fft.fftfreq(x.shape[0])), np.abs(x), color='blue')
        plt.axvline(2/365, color='r', linestyle='--', label=f'{np.fft.fftshift(np.fft.fftfreq(x.shape[0]))[np.argmax(x)]}')
        plt.xlabel('f')
        plt.ylabel('S(f)')
        plt.title(f'FFT of $x̃_{i}$')
        plt.legend()
        plt.show()
        print(f'The dominant frequency is at {np.fft.fftshift(np.fft.fftfreq(x.shape[0]))[np.argmax(x)]}')

        plt.plot(U_temporal[:, i], color='blue')
        plt.title(f'{i}th Temporal Principal component')
        plt.xlabel('time')
        plt.ylabel('PC value')
        plt.show()

        # perform fft, shift to be in order for the plot. take absolute value to find magnitude of peaks
        x = np.fft.fftshift(np.fft.fft(U_temporal[:, i] - np.mean(U_temporal[:, i])))
        plt.plot(np.fft.fftshift(np.fft.fftfreq(x.shape[0])), np.abs(x), color='blue')
        plt.axvline(2/365, color='r', linestyle='--', label=f'{2/365}')
        plt.xlabel('f')
        plt.ylabel('S(f)')
        plt.title(f'FFT of temporal Principal Component {i}')
        plt.legend()
        plt.show()
        print(f'The dominant frequency is at {np.fft.fftshift(np.fft.fftfreq(x.shape[0]))[np.argmax(x)]}')

    return None #modify if needed


#===== Code for Part 2=====#
def part2(f,method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m,n = f.shape
    fI = np.zeros((m-1,n)) #use/modify as needed

    if method==1:
        fI = 0.5*(f[:-1,:]+f[1:,:])
    else:
        #Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1
        
        #coefficients for near-boundary points
        a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)

        # constructing b
        M = np.diag(a/2* np.ones(m)) + np.diag(a/2 * np.ones(m-1), k=1) + np.diag( b/2 * np.ones(m-1), k=-1) + np.diag(b/2* np.ones(m-2), k=2)
        M[0, :4] = [a_bc,b_bc,c_bc,d_bc]
        M[-2, -4:] = [a_bc,b_bc,c_bc,d_bc][::-1]
        b = M[:-1, :]@f

        # construct banded matrix A
        alphab = alpha*np.ones(m-1)
        alphaa = alphab.copy()
        alphaa[0:2] = 0
        alphab[-2:] = 0
        onesd = np.ones(m-1)
        Ab = np.array([alphaa, onesd, alphab])
        # use existing solve methods for efficiency
        fI = scipy.linalg.solve_banded((1,1), Ab, b)
        
    return fI #modify as needed

def part2_analyze():
    """
    Add input/output as needed
    """

    #----- Code for generating grid, use/modify/discard as needed ----#
    n,m = 50,50 #arbitrary grid sizes - set to m=n
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    #--------------------------------------------#

    for method in np.arange(1, 3): # for both methods
        norms = []
        for k in np.arange(800):
            f_yI = np.exp(1j * k * yI) # true values under the function
            f_yg = np.exp(1j * k * yg)  # observed valuyes under function
            f_fI = part2(f_yg, method=method) # interpolated values
            norms.append(np.linalg.norm(f_yI - f_fI[:,0])) # mse

        plt.figure()
        plt.plot(norms, color='blue')
        plt.title(f'Error at wave number k, for method {method}')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.show()

    alpha = 0.3
    a = 1.5
    b = 0.1
    kh_max = [4, 50] # to demonstrate the periodicity of the modified wave number
    for k_max in kh_max:
        kh_range = np.arange(0.01, k_max, 0.01)

        m1_list = []
        m2_list = []
        for x in kh_range:
            m1_list.append(x * np.cos(x /2))
            m2_list.append(x * (b * np.cos(3*x/2) + a * np.cos(x/2) - alpha * 2 * np.cos(x))) 

        # finding where 1% error exists, determine how many gridpoints per wavelength this corresponds to
        index_m1 = np.where(((kh_range - m1_list) / kh_range) >= 0.01)[0][0]
        kh_m1 = kh_range[index_m1]
        lamh_m1 = ceil(2 * np.pi/kh_m1) # take the ceiling to make sure whole number

        index_m2 = np.where((kh_range - m2_list / kh_range) >= 0.1)[0][0]
        kh_m2 = kh_range[index_m2]
        lamh_m2 = ceil(2*np.pi/kh_m2)  # take the ceiling to make sure whole number

        plt.plot(kh_range, kh_range, label='kh', color='purple')
        plt.plot(kh_range, m1_list, label='method 1', color='red')
        plt.plot(kh_range, m2_list, label='method 2', color='blue')
        plt.plot(kh_range[index_m1], m2_list[index_m1], '.', color='yellow', label='Error for method 1 ≈ 1%')
        plt.plot(kh_range[index_m2], m2_list[index_m2], '.', color='lime', label='Error for method 2 ≈ 1%')
        plt.xlabel('kh')
        plt.ylabel('modified kh')
        plt.title('Modified wavenumber over range of kh values')
        plt.legend()
        plt.show()

    print(f'The number of points per wavelength for 1% error in method 1 is {lamh_m1}, and in method 2 is {lamh_m2}')

    # use the gridpoints per wavelength determines to then plot the error at each index for fixed x
    n,m = 23,23
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2
    k=20
    denoms = [60, 220]
    for denom in denoms:
        h = np.pi/denom
        n,m = h,h
        x = np.arange(0,1,n)
        y = np.arange(0,1,m)
        xg,yg = np.meshgrid(x,y)
        dy = y[1]-y[0]
        yI = y[:-1]+dy/2
        norms1 = [] 
        norms2 = []
        for i in np.arange(y.shape[0]-1):
            f_yI = np.exp(1j * k * yI) # true values under the function
            f_yg = np.exp(1j * k * yg) 
            f_fI_1 = part2(f_yg, method=1) # interpolated values
            norms1.append(f_yI[i] - f_fI_1[0,i])
            f_fI_2 = part2(f_yg, method=2) # interpolated values
            norms2.append(f_yI[i] - f_fI_2[0,i])

        plt.plot(y[:-1], np.abs(norms2), color='red', label='method 2')
        plt.plot(y[:-1], np.abs(norms1), color='blue', label='method 1')
        plt.title(f'Errors for {int(denom/10)} gridpoints per wavelength')
        plt.legend()
        plt.xlabel('$y$')
        plt.ylabel('$y_{true} - y_{interpolated}$')

        plt.show()


    return None #modify as needed



#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray


def part3_analyze(display = False):#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000
    cs = [0.5, 1.0, 1.2, 1.3, 1.4, 1.5] # the cs to contour plot

    for c in cs:

        #Set initial conidition
        L = (n-1)/np.sqrt(beta)
        k = 40*np.pi/L
        a0 = np.linspace(0,L,n)
        A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
        y0 = np.zeros(2*n)
        y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

        t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') # not removing much of transient here
        y0 = y[-1,:]
        t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
        u=y[:,100:n-100]


        if display:
            plt.figure()
            plt.contourf(np.arange(n-200),t,u,20)
            plt.title(f'Contour plot of u at c={c}')
            plt.show()

        if c == 0.5:
            # find dominant frequency using welch (lab9sols)
            from scipy.signal import welch
            fxx,Pxx = welch(u.T,fs=1/t[1])
            plt.contourf(np.arange(3800), fxx, Pxx.T, 20)
            plt.colorbar()  
            plt.xlabel('i')
            plt.ylabel('Frequency')
            plt.title(f'Contour plot to show dominant frequencies at c={c}')
            plt.show()

            # take PCA, plotting cumulative variance
            A = np.transpose(u - np.mean(u, axis=0))
            U, _, _ = np.linalg.svd(A.T @ A)
            Cov = 1/(u.shape[1]-1)*A.dot(A.T)
            total_variance = np.sum(np.diag(Cov))
            evals, _ = np.linalg.eig(Cov)
            plt.plot((np.cumsum(np.abs(evals))/total_variance)[:100],color='blue')
            plt.title('Variance explained by the first 100 Principal Components, c=0.5')
            plt.xlabel('PC')
            plt.ylabel('Cumulative Variance')
            plt.show()
            
            plt.plot(U[:, 0], color='blue')
            plt.xlabel('i')
            plt.ylabel('$PC_i$')
            plt.title('First Principal Component when c=0.5')
            plt.show()

        if c == 1.3:
            from scipy.spatial.distance import pdist
            # remove transient states this time
            y0 = np.zeros(2*n)
            y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real
            t,y = part3q1(y0,alpha,beta,b,c,tf=200,Nt=2,method='RK45') # more time steps to get to transient
            y0 = y[-1,:]
            t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
            ut=y[:,100:n-100]

            plt.figure()
            plt.contourf(np.arange(n-200),t,ut,20)
            plt.title(f'Contour plot of u at c={c}, transient states removed')
            plt.show()

            fxx,Pxx = welch(ut.T,fs=1/t[1])
            plt.contourf(np.arange(3800), fxx, Pxx.T, 20)
            plt.colorbar()
            plt.xlabel('i')
            plt.ylabel('Frequency')
            plt.title(f'Contour plot to show dominant frequencies at c={c}, with transient states removed')
            plt.show()

            # for fractal dimension analysis, compute correlation sum
            Dt = pdist(ut)
            Ct = []
            epsilonst = np.linspace(min(Dt)+1, max(Dt), 200)[::-1] # reverse epsilons
            for epsilon in epsilonst:
                Dt = Dt[Dt<epsilon]
                Ct.append(Dt.size/comb(3800, 2))

            D = pdist(u)
            C = []
            for epsilon in epsilonst:
                D = D[D<epsilon]
                C.append(D.size/comb(3800, 2))

            plt.plot(np.log(epsilonst), np.log(C), label='Before cutting transient states')
            plt.plot(np.log(epsilonst), np.log(Ct), label = 'Transient state cut')
            grad, intercept = np.polyfit(np.log(epsilonst[-50:-20]), np.log(C[-50:-20]), deg=1) # find the gradient in the middle
            yfit = np.poly1d([grad, intercept]) # plot with gradient, but consider that it is a loglog so convert intercept and gradient approrpiately
            plt.plot(np.log(epsilonst[-50:-20]),np.log(np.exp(intercept)*np.power(epsilonst[-50:-20],grad)), label='slope = '+str(grad), color='lime')
            grad1, intercept1 = np.polyfit(np.log(epsilonst[85:135]), np.log(Ct[85:135]), deg=1)
            yfit = np.poly1d([grad1, intercept1])
            plt.plot(np.log(epsilonst[85:135]),np.log(np.exp(intercept1)*np.power(epsilonst[85:135],grad1)), label='slope = '+str(grad1))
            plt.legend()
            plt.xlabel('ln(epsilon)')
            plt.ylabel('ln(Corrlation Sum)')
            plt.show()

            # perform PCA on the data with transient states removed c=1.3, same as above
            A = np.transpose(ut - np.mean(ut, axis=0))
            U, _, _ = np.linalg.svd(A.T @ A)
            Cov = 1/(u.shape[1]-1)*A.dot(A.T)
            total_variance = np.sum(np.diag(Cov))
            evals, _ = np.linalg.eig(Cov)
            plt.plot((np.cumsum(np.abs(evals))/total_variance)[:100],color='blue')
            plt.title('Variance explained by the first 100 Principal Components, c=1.3')
            plt.xlabel('PC')
            plt.ylabel('Cumulative Variance')
            plt.show()
            
            plt.plot(U[:, 0], color='blue')
            plt.xlabel('i')
            plt.ylabel('$PC_i$')
            plt.title('First Principal Component when c=1.3')
            plt.show()
    #-------------------------------------------#

    #Add code here


    return None #modify if needed

def part3q2(x,c=1.0):
    """
    Code for part 3, question 2
    """
    #Set parameters
    y0 = np.zeros(2*n)
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A))
    v2 = A.dot(v1)
    A2 = (v2[:,:x]).dot((v1[:,:x]).T)
    e = np.sum((A2.real-A)**2)

    return A2.real,e


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed
