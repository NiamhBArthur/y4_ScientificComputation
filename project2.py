"""
Code for Scientific Computation Project 2
Please add college id here
CID: 01864994
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
#use scipy in part 2 as needed

#===== Codes for Part 1=====#
def searchGPT(graph, source, target):
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('inf') for node in graph}
    distances[source] = 0

    # Initialize a priority queue to keep track of nodes to explore
    priority_queue = [(0, source)]  # (distance, node)

    # Initialize a dictionary to store the parent node of each node in the shortest path
    parents = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            path.insert(0, source)
            return current_distance,path

        # If the current distance is greater than the known distance, skip this node
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = max(distances[current_node], weight['weight'])
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')  # No path exists


def searchPKR(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s])
    Mdict[s]=Mlist[0]
    found = False

    while len(Mlist)>0:
        dmin,nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin
        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp,en]
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    return dmin


def searchPKR2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to searchGPT given same input
    """

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s,[s]]) # initialise Mlist with the source node as the current path
    Mdict[s]=Mlist # initialise Mdict
    found = False

    while len(Mlist)>0: # while the priority list has elements
        dmin,nmin,path = heapq.heappop(Mlist) # pop the first element (weight, end node, path)
        if nmin == x: # if we have found the target, found is true
            found = True
            break # leave the while loop if target is found
        
        Fdict[nmin] = dmin # set up the finished dictionary
        for m,en,wn in G.edges(nmin,data='weight'): 
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]: # smaller weight? update the current data for this node
                    l = Mdict.pop(en)
                    l[1] = n
                    lnew = [dcomp,en,path +[en]] # put the neighbour at the end of the current path
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en,path +[en]] # put the neighbout at the end of the current path
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    if found == True:
        return dmin, path # return the weight and path if target has been found
     
    return float('inf') # else return infinity

#===== Code for Part 2=====#
def part2q1(y0,tf=1,Nt=5000):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        for i in range(1,n-1):
            dydt[i] = alpha*y[i]-y[i]**3 + beta*(y[i+1]+y[i-1])
        
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 


    #Compute numerical solutions
    dt = tarray[1]
    for i in range(Nt):
        yarray[i+1,:] = yarray[i,:]+dt*RHS(0,yarray[i,:])

    return tarray,yarray

def part2q1new(y0,tf=40,Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    # initialise the matrix
    

    def RHS(t,y):
        """
        Compute RHS of model
        """        

        n = y.shape[0]
        betas = beta * np.ones(n-1) # initialise the off diagonals
        alphas = alpha * np.ones(n) # initialise the diagonal
        M = np.diag(alphas, 0) + np.diag(betas, 1) + np.diag(betas,-1) # add the diagonals to form the matrix
        M[0,-1] = beta # replace corners
        M[-1,0] = beta
        dydt = y@M - np.power(y, 3*np.ones(n)) # matrix multiplication and vector subtraction to reduce FLOPS

        return dydt 
    
    # use solve_ivp to improve efficiency
    sol = scipy.integrate.solve_ivp(fun=RHS, t_span=(0, tf), y0=y0, t_eval=np.linspace(0, tf, Nt+1), method='BDF')
    
    return sol.t, sol.y.T


def part2q2(): #add input variables if needed
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load('project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition
    beta = 10000 / np.pi**2
    alpha = 1-2*beta
    n = y0A.size

    def jac(y):
        """
        Compute the jacobian at point y
        """

        n=y.shape[0]

        betas = beta * np.ones(n-1) # initialise diagonals
        alphas = alpha * np.ones(n) - 3* y**2
        jac = np.diag(alphas, 0) + np.diag(betas, 1) + np.diag(betas,-1) # add diagonals to form matrix
        jac[0,-1] = beta
        jac[-1,0] = beta
        return jac
        
    def RHS(y):
        """
        Compute RHS of model
        """        

        n = y.shape[0]
        betas = beta * np.ones(n-1) # initialise the off diagonals
        alphas = alpha * np.ones(n) # initialise the diagonal
        M = np.diag(alphas, 0) + np.diag(betas, 1) + np.diag(betas,-1) # add the diagonals to form the matrix
        M[0,-1] = beta # replace corners
        M[-1,0] = beta
        dydt = y@M - np.power(y, 3*np.ones(n)) # matrix multiplication and vector subtraction to reduce FLOPS

        return dydt 


    # determine the evolutions of both initial conditions

    tarrayA, yarrayA = part2q1new(y0A, tf=40, Nt=1000)
    tarrayB, yarrayB = part2q1new(y0B, tf=40, Nt = 1000)

    # find the true solutions around the initial conditions

    solA = scipy.optimize.root(RHS, y0A).x
    solB = scipy.optimize.root(RHS, y0B).x

    # making the plots for part2q2

    plt.figure()
    plt.plot(tarrayA, yarrayA)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Evolution from IC A')
    plt.show()

    plt.figure()
    plt.plot(tarrayB, yarrayB)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Evolution from IC B')
    plt.show()

    # determine the eigenvalues for part2q2 and part2q3

    eigs1 = scipy.linalg.eigh(jac(np.ones(3)))[0]
    eigs2 = scipy.linalg.eigh(jac(-1*np.ones(3)))[0]
    eigs3 = scipy.linalg.eigh(jac(np.zeros(3)))[0]

    eigsA = scipy.linalg.eigh(jac(solA))[0]
    eigsB = scipy.linalg.eigh(jac(solB))[0]
    eigsC = scipy.linalg.eigh(jac(np.ones(1000)))[0]

    # find which eigenvalues are non negative

    poseigs1 = [i for i in eigs1 if i > 0]
    poseigs2 = [i for i in eigs2 if i > 0]
    poseigs3 = [i for i in eigs3 if i > 0]

    poseigsA = [i for i in eigsA if i > 0]
    poseigsB = [i for i in eigsB if i > 0]
    poseigsC = [i for i in eigsC if i > 0]

    # print non negative eigenvalues

    print('The positive eigenvalues for solution of 1s is', poseigs1)
    print('The positive eigenvalues for solution of -1s is', poseigs2)
    print('The positive eigenvalues for solution of 0s is', poseigs3)

    print('The positive eigenvalues for solution of A is', poseigsA)
    print('The positive eigenvalues for solution of B is', poseigsB)
    print('The positive eigenvalues for solution of 1s (in 1000 dimensions) is', poseigsC)
 
    return None


def part2q3(tf=10,Nt=1000,mu=0.2,seed=1):
    """
    Input:
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same random numbers are generated with each simulation

    Output:
    tarray: size Nt+1 array
    X size n x Nt+1 array containing solution
    """

    #Set initial condition
    y0 = np.array([0.3,0.4,0.5])
    np.random.seed(seed)
    n = y0.size #must be n=3
    Y = np.zeros((Nt+1,n)) #may require substantial memory if Nt, m, and n are all very large
    Y[0,:] = y0

    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    beta = 0.04/np.pi**2
    alpha = 1-2*beta

    def RHS(t,y):
        """
        Compute RHS of model
        """
        dydt = np.array([0.,0.,0.])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[2])
        dydt[1] = alpha*y[1]-y[1]**3 + beta*(y[0]+y[2])
        dydt[2] = alpha*y[2]-y[2]**3 + beta*(y[0]+y[1])

        return dydt 

    dW= np.sqrt(Dt)*np.random.normal(size=(Nt,n))

    #Iterate over Nt time steps
    for j in range(Nt):
        y = Y[j,:]
        F = RHS(0,y)
        Y[j+1,0] = y[0]+Dt*F[0]+mu*dW[j,0]
        Y[j+1,1] = y[1]+Dt*F[1]+mu*dW[j,1]
        Y[j+1,2] = y[2]+Dt*F[2]+mu*dW[j,2]

    return tarray,Y


def part2q3Analyze(): #add input variables as needed
    """
    Code for part 2, question 3
    """

    # find trajectories for various values of mu

    _, Y20 = part2q3(mu=0, tf=70)
    _, Y2 = part2q3(mu=0.2, tf=70)
    _, Y2A = part2q3(mu=0.55, tf=70)

    # initialise lists for the plots
    colors=['orange', 'gold', 'goldenrod']
    labels = ['$\mu = 0.55$', '_','_']

    # code for figure 2

    plt.figure()
    for i in range(3):
        plt.plot(Y2A[:, i], color=colors[i], label=labels[i])
    plt.plot(Y2, color='hotpink', label=['$\mu = 0.2$', '_','_'])
    plt.plot(Y20, color='darkblue', label=['$\mu = 0$', '_','_'])
    plt.legend()
    plt.xlabel('N$_t$')
    plt.ylabel('y')
    plt.title('Plot of solutions when $\mu=0$ vs. $\mu=0.2$ vs. $\mu=0.55$')
    plt.show()
    


    return None 


