"""
Code for Scientific Computation Project 1
Please add college id here
CID: 01864994
"""

from time import time
import matplotlib.pyplot as plt
import numpy as np

Xtest = np.random.randint(0, 1000, 1000)

#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy() 
    for i,x in enumerate(X[1:],1):
        if i<=istar:
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a
        
        X[ind+1:i+1] = X[ind:i]
        X[ind] = x

    return X


def part1_time(fig1=False, fig2=False, fig3=False, fig4=False):
    """Examine dependence of walltimes of part1 function on N and istar
        You may modify the input/output as needed.
        Set figx to true for the figure from the pdf to be created.
    """
    def wall_time(N, istar, X):
        t1 = time() # find the time before the function is called
        part1(X[:N], istar) # run the function
        t2 = time() # find the time after the function is called

        return t2-t1 # return the difference in times
    
    Xrev = np.arange(1000, 1, -1) # making a descending list, the worst case

    times_istar1000 = [] # initialise the list
    for i in range(1000): 
        times_istar1000.append(wall_time(1000, i, Xrev)) # for N=1000, loop over possible istar values to determine optimal istar at N=1000

    times_0 = []   # initialise the list
    for i in range(1000):
        times_0.append(wall_time(i, 0, Xrev)) # for N up to 1000, time when istar = 0

    times_n1 = [] # initialise the list
    for i in range(1000):
        times_n1.append(wall_time(i, i-1, Xrev)) # for N up to 1000, time when istar = N-1

    times_n2 = [] # initialise the list
    for i in range(1000):
        times_n2.append(wall_time(i, i/2, Xrev)) # for N up to 1000, time when istar = N/2

    times_sqrt = [] # initialise the list
    for i in range(1000):
        times_sqrt.append(wall_time(i, i**(1/2), Xrev)) # for N up to 1000, time when istar = sqrt(N)

    times_min = [] # initalise the list
    for i in range(1000):
        times_min.append(wall_time(i, np.argmin(times_istar1000), Xrev)) # for N up to 1000, time when istar = min istar from previous istar tests

    # print(np.argmin(times_istar1000))  # uncomment to find the min istar for this run

    if fig1:
        plt.figure()
        plt.loglog(times_0, label='istar = 0')
        plt.loglog(times_n1, label='istar = N-1')
        plt.loglog(times_n2, label='istar = N/2')   
        plt.xlabel('N')
        plt.ylabel('time')
        plt.legend()
        plt.title('Times of part1 for various istar, over all N')
        plt.show()

    if fig2:
        poly0 = np.polyfit(range(1000), times_0, deg=2)
        polyn1 = np.polyfit(range(1000), times_n1, deg=2)
        polyn2 = np.polyfit(range(1000), times_n2, deg=2)

        plt.plot(times_0, label='istar = 0')
        plt.plot(times_n1, label='istar = N-1')
        plt.plot(times_n2, label='istar = N/2')
        plt.plot(np.poly1d(poly0)(range(1000)), label='quadratic fit to istar = 0')
        plt.plot(np.poly1d(polyn1)(range(1000)), label='quadratic fit to istar = N-1')
        plt.plot(np.poly1d(polyn2)(range(1000)), label='quadratic fit to istar = N/2')
        plt.title('Times of part1 function for various istar, over all N, with quadratics fit')
        plt.legend()
        plt.xlabel('N')
        plt.ylabel('time')
        plt.show()

    if fig3:
        plt.plot(times_0, label='istar = 0')
        plt.plot(times_min, label='istar = argmin time when N=1000')
        plt.plot(times_sqrt, label='istar = sqrt(N)')
        plt.legend()
        plt.title('Times of part1 function for various istar, over N')
        plt.xlabel('N')
        plt.ylabel('time')
        plt.show()
    

    if fig4:
        plt.plot(times_istar1000)
        plt.title('Times for each value of istar when N=1000')
        plt.xlabel('istar')
        plt.ylabel('time')
        plt.show()


part1_time(fig3=True)



#===== Code for Part 2=====#

def part2(S,T,m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
   """
    #Size parameters
    n = len(S) 
    l = len(T) 
    # base parameters for Rabin-Karp
    Base = 4 # due to 4 letters
    Prime = 2**24 -1 # large enough prime to avoid overlaps
    q = Prime
    

    # L = [[j for j in range(n-m+1) if T[i:i+m-1] == S[j:j+m-1]] for i in range(l-m+1)] # naive search

    # function to convert into string into list of integers
    def char2base4(S):
        """Convert gene test_sequence string to list of ints
        """
        c2b = {}
        c2b['A']=0
        c2b['C']=1
        c2b['G']=2
        c2b['T']=3
        L=[]
        for s in S:
                L.append(c2b[s])

        return L
            
    # function to convert a list into the initial hash value
    def heval(L,Base,Prime):
        """Convert list L to base-10 number mod Prime where Base specifies the base of L
        """
        f=0
        for l in L[:-1]:
            f = Base*(l+f)
        h = (f + (L[-1])) % Prime
        return h
    
    S = char2base4(S) # convert our lists appropriately
    T = char2base4(T)
    imatches = [] # initialise the final list
    dict = {} # initialise the hash map 
    X = S # change notation to follow code adapted from lecture notes

    for ind in range(l-m+1): # loop over sequence T
            
        Y = T[ind:ind+m] # for each combination of T
        hp = heval(Y,Base,Prime) # work out the hash we are looking for

        if hp in dict: # if the hash is in the dictionary already then append the index
            # likely need to string check here, however time dependent
            dict[hp].append(ind)

        dict[hp] = [ind] # otherwise set up a new key and value pair

        imatches.append([]) # initialising the list of lists for each index
    
    hi = heval(X[:m],Base,Prime)  # initialise the rolling hash throughout S
    if hi in dict: # check the first hash is in dict (hash collision)
        for i in dict[hi]: # for each index listed in the dict
            if X[:m] == T[i: i+m]:  # check that the strings actually match
                imatches[i].append(0) # append the first index

        
    bm = (4**m) % q # initialise for rolling hash function
    for ind in range(1,n-m+1): # loop over sequence S
        #Update rolling hash
        hi = (4*hi - int(X[ind-1])*bm + int(X[ind-1+m])) % q
        # check that new hash is in the hashmap
        if hi in dict:
            for i in dict[hi]: # check all indices paired with the hash
                if X[ind:ind+m] == T[i: i+m]: # check strings actually match
                    imatches[i].append(ind) # append the index appropriately
        

    return imatches # return the list of lists


if __name__=='__main__':
    #Small example for part 2
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    

    #Large gene sequence from which S and T test sequences can be constructed
    infile = open("test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()

    out = part2(S,T,m)
