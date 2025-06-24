"""
Project 4 code
CID: 01864994
"""

import numpy as np
import matplotlib.pyplot as plt
import time as time
#use scipy as needed


def load_image(normalize=True,display=False):
    """"
    Load and return test image as numpy array
    """
    import scipy
    from scipy.datasets import face
    A = face()
    if normalize:
        A = A.astype(float)/255
    if display:
        plt.figure()
        plt.imshow(A)
    return A


#---------------------------
# Code for Part 1
#---------------------------
def delta_truncated_svd(matrix, delta):
    """
    Compute delta-truncated SVD

    Parameters:
    - matrix: Input matrix to perform trundacted SVD on.
    - delta: Maximum error.

    Returns:
    - U: U from SVD
    - S: Diagnonal entries of S from SVD
    - Vt: V transposed from SVD

    """

    # Compute SVD on the matrix
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate the matrices based on delta
    cum_var = np.cumsum(S[::-1] ** 2)[::-1] # cumsum descending orderm- changes when squared
    trunc_var = delta ** 2
    k = cum_var > trunc_var # take index values where the variance is explained

    U = U[:, k]
    S = S[k]
    Vt = Vt[k, :]

    return U, np.diag(S), Vt

def decompose1(A,eps):
    """
    Implementation of Algorithm 3 from KCSM
    Input:
    A: tensor stored as numpy array 
    eps: accuracy parameter
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    # initialisations
    idxs = A.shape
    N = len(idxs)
    R_list = np.zeros(N, dtype=np.int64)
    R_list[0] = 1 # initial rank to 1
    Glist = []
    
    # determine delta and initialise zZ.
    delta = eps / np.sqrt(N-1) * np.linalg.norm(A) # defaults to frobenius norm
    Z = A.reshape((idxs[0], -1)) # unfold

    # loop over each of the modes to determine the tensor cores
    for n in range(N-1):
        U, S, V = delta_truncated_svd(Z, delta)
        R_list[n+1] = np.size(U, 1)
        Glist.append(U.reshape((R_list[n], idxs[n], R_list[n+1])))
        Z = np.reshape(S@V, (R_list[n+1]*idxs[n+1], -1))

    # find the final tensor core using Z.
    Glist.append(Z.reshape(R_list[n+1], idxs[n+1], 1))
    return Glist

def reconstruct(G):
    """
    Reconstruct a tensor given its tensor-train form.
    Input G: List of tensor cores
    Output A_finished: Reonstructed tensor
    """
    # initialise
    reshape_idxs = []
    A_reconstructed = G[0]

    # tensordot along the tensor train
    for i in range(1, len(G)):
        A_reconstructed = np.tensordot(A_reconstructed, G[i], axes=([-1], [0]))
    
    # determine the shape of the reconstruction 
    for i in range(1, len(G)):
        reshape_idxs.append(A_reconstructed.shape[i])
    reshape_idxs.append(-1)
    # reshape to be the same size as original tensor
    A_finished = A_reconstructed.reshape(reshape_idxs)
    return A_finished


def decompose2(A,Rlist, compression_rate=False):
    """
    Implementation of modified Algorithm 3 from KCSM with rank provided as input
    Input:
    A: tensor stored as numpy array 
    Rlist: list of values for rank, [R1,R2,...]
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """
    # initialise
    Glist = []
    idxs = A.shape
    N = len(idxs)
    Rlist.insert(0, 1)
    Rlist.append(1)
    Z = A.reshape((idxs[0], -1))
    if compression_rate:
        start = time.time()

    # loop over each of the modes
    for n in range(N-1):
        U, S, V = np.linalg.svd(Z, full_matrices=False)
        U = U[:, :Rlist[n+1]]
        S = np.diag(S[:Rlist[n+1]])
        V = V[:Rlist[n+1], :]
        Glist.append(U.reshape((Rlist[n], idxs[n], Rlist[n+1])))
        Z = np.reshape(S@V, (Rlist[n+1]*idxs[n+1], -1))
    # determine the final tensor core
    Glist.append(Z.reshape(Rlist[n+1], idxs[n+1], 1))

    # test values for plots
    if compression_rate:
        end = time.time()
        timer = end - start
        rate = 0
        for g in Glist:
            rate += np.prod(np.array(g.shape))
        rate /= np.prod(np.array(idxs))
        return Glist, rate, timer

    return Glist




def part1():
    """
    Add code here for part 1, question 2 if needed
    """

    # code to print raccoon image for decompose1 and decompose2

    A = load_image()

    # work out relative residual error
    def recompose_error(A, A_new):
        return np.linalg.norm((A - A_new))/np.linalg.norm(A)

    dec2, _, _ = decompose2(A, [10, 3], compression_rate=True)
    errord2 = recompose_error(A, reconstruct(dec2))

    dec1 = decompose1(A, 0.26)
    errord1 = recompose_error(A, reconstruct(dec1))

    plt.figure()
    plt.imshow(reconstruct(dec2))
    plt.title(f'Recomposition with decompose2, error={errord2}')
    plt.show()
    plt.figure()
    plt.imshow(reconstruct(dec1))
    plt.title(f'Recomposition with decompose1, error={errord1}')
    plt.show()

    return None #modify as needed


#-------------------------
# Code for Part 2
#-------------------------
def part2(figures=False, long_run_time=False):
    """
    Note: if you set long_run_time=True, some aspects of this function take over an hour to run.
    """
    # load image and video
    A = load_image()
    V = video2numpy('project4.mp4')


    def mode_n_prod(G, A, mode):
        """
        Calulate the mode-n-product
        Input G: Tensor to calculate product on
        Input A: Matrix to calculate product on
        Input mode: Specify along which mode
        Output product: returns mode n product
        """

        # ensure shape and reshaping is correct
        orig_shape = list(G.shape)
        new_shape = orig_shape
        new_shape[mode] = A.shape[0]
        # flatten G
        G_flat = np.reshape(np.moveaxis(G, mode, 0), (G.shape[mode], -1))
        shape_i = list(new_shape).copy()
        mode_dim = shape_i.pop(mode)
        shape_i.insert(0, mode_dim)
        # multiply and fold G again into its product
        product = np.moveaxis(np.reshape(A @ G_flat, (shape_i)), 0, mode)

        return product

    def hosvd(T, ranks, compression_rate=False):
        """
        Perform HOSVD decomposition
        Input T: Tensor to decompose
        Input ranks: over which ranks
        Input compression_rate: Boolean for outputs to create figure
        Output core: core tensor in tucker form
        Output factors: factor matrices in tucker form
        Output rate: compression rate of decomposition
        Output total_time: time taken to perform algorithm
        """
        start = time.time()
        shape = np.array(T.shape)
        factors = []
        core = T.copy()

        for mode, rank in enumerate(ranks):
            # Mode-n unfolding of the tensor
            T_flat = np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))

            # Perform SVD on the unfolded tensor
            U, _, _ = np.linalg.svd(T_flat, full_matrices=False)

            # Truncate the factor matrix to the desired rank
            factors.append(U[:, :rank])
            # print(core.shape, U[:, :rank].shape)
            
            core = mode_n_prod(core, factors[mode].T, mode)
        if compression_rate:
            end = time.time()
            rate = np.dot(shape, ranks) + np.prod(ranks)
            rate /= np.prod(shape)
            total_time = end-start
            return core, factors, rate, total_time

        return core, factors

    def hosvd_recompose(G, A_list):
        """
        Recompose a tensor from its core tensor and factor matrices obtained through HOSVD.

        Parameters:
        - G: Core tensor in the Tucker decomposition.
        - A_list: List of factor matrices.

        Returns:
        - tensor_recomposed: Recomposed tensor.
        """

        # mode n product of core tensor against each of the factor matrices
        for mode, factor in enumerate(A_list):
            G = mode_n_prod(G, factor, mode)

        return G

    
    def low_rank(T, ranks, compression_rate=False):
        """
        Repeated low-rank matrix approximations for tensor decomposition
        Input T: tensor to decompose
        Input ranks: specify truncation point at each layer
        Input compression_rate: Boolean for outputs to create figure
        Output low_rank_U: truncated U from SVD decomposition
        Output low_rank_S: truncated S from SVD
        Output low_rank_V: truncated V transpose from SVD
        Output rate: compression rate of decomposition
        Output timer: time taken to perform algorithm

        """
        # initialise
        low_rank_U = []
        low_rank_S = []
        low_rank_V = []
        rate=0

        if compression_rate:
            start = time.time()
        
        # for each layer, perform PCA
        for dim in range(T.shape[-1]):
            rank = ranks[dim]
            temp_matrix = T[:, :, dim]
            U, S, V = np.linalg.svd(temp_matrix, full_matrices=False)
            low_rank_U.append(U[:, :rank])
            low_rank_S.append(S[:rank])
            low_rank_V.append(V[:rank, :])

        if compression_rate:
            # make graph variables
            end = time.time()
            timer = end - start
            for rank in ranks:
                rate += rank*(T.shape[0]+T.shape[1]+1)
            rate /= np.prod(np.array(T.shape))
            return low_rank_U, low_rank_S, low_rank_V, rate, timer
            
        return low_rank_U, low_rank_S, low_rank_V

    def reconstruct_low_rank(low_rank_U, low_rank_S, low_rank_V):
        """
        Function to reconstruct the low-rank approximations method
        Input low_rank_U: U from SVD
        Input low_rank_S: S from SVD
        Input low_rank_V: V from SVD
        Output recomposed: Reconstructed tensor
        """
        low_rank_A = []
        # reform each of the layer matrices
        for i in range(len(low_rank_U)):
            low_rank_A.append(low_rank_U[i] @ np.diag(low_rank_S[i]) @ low_rank_V[i])

        # stack layers on top of each other
        recomposed = np.dstack(low_rank_A)
        return recomposed

    
    def recompose_error(A, A_new):
        """
        Function to determine the relative residual error
        """
        return np.linalg.norm((A - A_new))/np.linalg.norm(A)
    
    def create_heatmap(A, R1_list, R2_list, HOSVD_case=0, low_rank_case=0):
        """
        Function to collect data for plots
        """
        rates = [] 
        errors = []
        times = []
        
        if low_rank_case:
            heatarray = np.zeros((len(R2_list), len(R1_list), len(low_rank_case)))
        else:
            heatarray = np.zeros((len(R2_list), len(R1_list)))
        
        for j, num2 in enumerate(R2_list):
            for i, num1 in enumerate(R1_list):
                if HOSVD_case:
                    G, Alist, rate, timet1 = hosvd(A, [num1, num2, HOSVD_case], compression_rate=True)
                    A_new = hosvd_recompose(G, Alist)
                    error = recompose_error(A, A_new)

                elif low_rank_case:
                    rate = np.zeros(len(low_rank_case))
                    error = np.zeros(len(low_rank_case))
                    for k, num3 in enumerate(low_rank_case):
                        U, S, V, rate_i, timet1 = low_rank(A, [num1, num2, num3], compression_rate=True)
                        A_new = reconstruct_low_rank(U, S, V)
                        rate[k] = rate_i
                        error[k] = recompose_error(A, A_new)
                        heatarray[j, i, k] = error[k]
                    
                else:
                    A_temp, rate, timet1 = decompose2(A, [num1, num2], compression_rate=True)
                    A_new = reconstruct(A_temp)
                    error = recompose_error(A, A_new)

                times.append(timet1)
                rates.append(rate)
                errors.append(error)

                if not low_rank_case:
                    heatarray[j, i] = errors[-1]
            
        return heatarray, rates, errors, times



    # Below, running code to collect data for figures.
    heatarray_list = []

    R1_list = range(100, 1020, 150)
    R2_list = range(100, 760, 100)
    rate_array = np.zeros((len(R1_list) * len(R2_list), 3))
    errors_array = np.zeros((len(R1_list) * len(R2_list), 3))
    time_array = np.zeros((len(R1_list) * len(R2_list), 3))
    for idx in range(1, 4):
        temp_heatarray, rates_tuck, errors_tuck, times_tuck = create_heatmap(A, R1_list, R2_list, HOSVD_case=idx)
        heatarray_list.append(temp_heatarray)
        rate_array[:, idx-1] = rates_tuck
        errors_array[:, idx-1] = errors_tuck
        time_array[:, idx-1] = times_tuck

    R1_list = range(1, 750, 60)
    R2_list = range(1, 4)
    heatarray_tt, rates_tt, errors_tt, times_tt = create_heatmap(A, R1_list, R2_list)

    R1_list = range(100, 750, 100)
    R2_list = range(100, 750, 100)
    R3_list = range(100, 750, 100)
    heatarray_low_rank, rates_low_rank, errors_low_rank, times_low_rank = create_heatmap(A, R1_list, R2_list, low_rank_case=R3_list)

    if long_run_time:
        # note: do not run this unless you have 3h to spare.
        # more data collection for functions
        R1_list = range(50, 100, 20)
        R2_list = range(100, 540, 200)
        R3_list = range(1, 4)
        temp1 = np.zeros((len(R1_list), len(R2_list), len(R3_list)))
        temp2 = np.zeros((len(R1_list), len(R2_list), len(R3_list)))
        temp3 = np.zeros((len(R1_list), len(R2_list), len(R3_list)))
        errors_video_ttsvd = np.zeros((len(R1_list), len(R2_list), len(R3_list)))
        rates_video_ttsvd = np.zeros((len(R1_list), len(R2_list), len(R3_list)))
        for k, idx3 in enumerate(R3_list):
            for j, idx2 in enumerate(R2_list):
                for i, idx1 in enumerate(R1_list):
                    vid_Glist, rates_videott, _ = decompose2(V, [idx1, idx2, idx3], compression_rate=True)
                    V_ttsvd = reconstruct(vid_Glist)
                    errors_video_ttsvd[i, j, k] = recompose_error(V, V_ttsvd)
                    rates_video_ttsvd[i, j, k] = rates_videott
                    temp1[i, j, k] = idx1
                    temp2[i, j, k] = idx2
                    temp3[i, j, k] = idx3


        R1_list = range(20, 100, 40)
        R2_list = range(100, 540, 200)
        R3_list = range(300, 960, 300)
        R4_list = range(2, 4)
        rates_video_hosvd = []
        errors_video_hosvd = []
        recompose_hosvd = []
        for idx4 in R4_list:
            for idx3 in R3_list:
                for idx2 in R2_list:
                    for idx1 in R1_list:
                        core_hosvd, factor_hosvd, rate_hosvd, _ = hosvd(V, [idx1, idx2, idx3, idx4], compression_rate=True)
                        V_hosvd = hosvd_recompose(core_hosvd, factor_hosvd)
                        errors_video_hosvd.append(recompose_error(V, V_hosvd))
                        rates_video_hosvd.append(rate_hosvd)
                        recompose_hosvd.append([idx1, idx2, idx3, idx4])


        ttsvd_ranks = [int(temp1[-3, -3, 2]), int(temp2[-3, -3, 2]), int(temp3[-3, -3, 2])]
        print_video_ttsvd, print_rate_ttsvd, _ = decompose2(V, ttsvd_ranks, compression_rate=True)
        hosvd_idxs = recompose_hosvd[-30]
        print_video_hosvd_core, print_video_hosvd_factors, print_rate_hosvd, _ = hosvd(V, hosvd_idxs, compression_rate=True)

    if figures:

        # code to plot figures

        plt.figure()
        plt.scatter(rates_low_rank, errors_low_rank)
        plt.xlabel('Compression Rate')
        plt.ylabel('Relative Error')
        plt.title('Relative error against compression rate for low-rank approx.')
        plt.axvline(1, linestyle='--',color='r')
        plt.show()

        plt.figure()
        plt.scatter(rates_tt[:13], errors_tt[:13], label='2nd rank = 1')
        plt.scatter(rates_tt[13:26], errors_tt[13:26], label='2nd rank = 2')
        plt.scatter(rates_tt[26:], errors_tt[26:], label='2rd rank = 3')
        plt.xlabel('Compression Rate')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.title('Relative error against compression rate for TTSVD')
        plt.axvline(1, linestyle='--',color='r')
        plt.show()

        # figure
        plt.figure()
        plt.imshow(heatarray_tt, cmap='viridis_r')
        plt.xticks(np.arange(len(R1_list)), R1_list)
        plt.xlabel('R1 Values')

        # Set y-axis ticks and labels
        plt.yticks(np.arange(len(R2_list)), R2_list)
        plt.ylabel('R2 Values')

        # Add colorbar for reference
        plt.colorbar(label='Recomposition Error')

        # Add a title if needed
        plt.title('Recomposition Error Heatmap')

        # Show the plot
        plt.show()

        # figure
        plt.figure()
        for i in range(3):
            plt.scatter(rate_array[:, i], errors_array[:, i], label=f'3rd rank = {i+1}')
        plt.xlabel('Compression Rate')
        plt.legend()
        plt.ylabel('Relative Error')
        plt.axvline(1, linestyle='--',color='r')
        plt.title('Relative error against compression rate for HOSVD')
        plt.show()

        # figure

        test = np.array(errors_tt[14:])
        np.argmax(test)
        indexes=list(range(14, 26))
        indexes2=list(range(27, 39))
        indexes.extend(indexes2)

        plt.figure()
        plt.scatter(rate_array[:, 1:], errors_array[:, 1:], label=f'HOSVD')
        plt.scatter(np.array(rates_tt)[indexes], np.array(errors_tt)[indexes], label='TTSVD')
        plt.scatter(rates_low_rank, errors_low_rank, label='low_rank')
        plt.xlabel('Compression Rate')
        plt.ylabel('Relative Error')
        plt.legend()
        plt.title('Relative error against compression rate for each of the functions')
        plt.axvline(1, linestyle='--',color='r')
        plt.show()

        # figure


        min = []
        max = []
        for heatarray in heatarray_list:
            min.append(np.min(heatarray))
            max.append(np.max(heatarray))

        ultimate_min = np.min(min)
        ultimate_max = np.max(max)

        for i, heatarray in enumerate(heatarray_list):
            plt.figure()
            plt.imshow(heatarray, cmap='viridis_r', vmin = ultimate_min, vmax= ultimate_max)
            plt.xticks(np.arange(len(R1_list)), R1_list)
            plt.xlabel('R1 Values')

            # Set y-axis ticks and labels
            plt.yticks(np.arange(len(R2_list)), R2_list)
            plt.ylabel('R2 Values')

            # Add colorbar for reference
            plt.colorbar(label='Recomposition Error')

            # Add a title if needed
            plt.title(f'Recomposition Error Heatmap, 3rd rank = {i+1}')

            # Show the plot
            plt.show()

        # figure
            
        lrraccoon1, lrraccoon2, lrraccoon3 = low_rank(A, [4, 35, 9])
        errorlr = recompose_error(A, reconstruct_low_rank(lrraccoon1, lrraccoon2, lrraccoon3)) 
        plt.figure()
        plt.imshow(reconstruct_low_rank(lrraccoon1, lrraccoon2, lrraccoon3))
        plt.title(f'Reconstructed image using low-rank, error={errorlr}')
        plt.show()

        raccoonhosvd1, raccoonhosvd2 = hosvd(A, [11, 11, 3])
        errorhosvd = recompose_error(A, hosvd_recompose(raccoonhosvd1, raccoonhosvd2))
        plt.figure()
        plt.imshow(hosvd_recompose(raccoonhosvd1, raccoonhosvd2))
        plt.title(f'Reconstructed image using HOSVD, error={errorhosvd}')
        plt.show()


        plt.figure()
        plt.scatter(times_tt, errors_tt, label='TTSVD')
        plt.scatter(time_array, errors_array, label='HOSVD')
        plt.scatter(times_low_rank, errors_low_rank, label='low-rank')
        plt.legend()
        plt.xlabel('Wall time')
        plt.ylabel('Relative error')
        plt.title('Wall time against relative error for each of the decompositions')
        plt.show()

    
        if long_run_time:
        # figure

            plt.figure()
            for i in range(3):
                plt.scatter(rates_video_ttsvd[:, :, i], errors_video_ttsvd[:, :, i], label=f'Colour dimension rank ={i+1}')
            plt.legend()
            plt.xlabel('Compression rate')
            plt.ylabel('Relative error')
            plt.title('Compression rate against relative error for the TTSVD decomposition')
            plt.show()

            rates_video_hosvd_new = np.array(rates_video_hosvd)
            errors_video_hosvd_new = np.array(errors_video_hosvd)

            colors = ['#ff7f0e', '#2ca02c']
            plt.figure()
            plt.scatter(rates_video_hosvd_new[:int(len(errors_video_hosvd)-1)//2 +1],errors_video_hosvd_new[:(len(errors_video_hosvd) - 1)//2+1], label='Colour rank=2', color=colors[0])
            plt.scatter(rates_video_hosvd_new[int(len(errors_video_hosvd)-1)//2+1:],errors_video_hosvd_new[(len(errors_video_hosvd) - 1)//2+1:], label='Colour rank = 3', color=colors[1])
            plt.legend()
            plt.xlabel('Compression rate')
            plt.ylabel('Relative error')
            plt.title('Compression rate against relative error for the TTSVD decomposition')
            plt.show()

            # figure 

            plt.figure()
            plt.scatter(rates_video_hosvd, errors_video_hosvd, label='HOSVD')
            plt.scatter(rates_video_ttsvd[:, :, 1:], errors_video_ttsvd[:, :, 1:], label='TTSVD')
            plt.xlabel('Compression rate')
            plt.ylabel('Relative error')
            plt.title('Compression rate against relative error for HOSVD and TTSVD')
            plt.legend()
            plt.show()

            # figure 
            frame_index = 82
            plt.figure()
            plt.imshow(reconstruct(print_video_ttsvd)[frame_index, :, :, :])
            plt.title(f'Reconstructed TTSVD image at compression rate = {print_rate_ttsvd}')
            plt.show()
            plt.figure()
            plt.imshow(hosvd_recompose(print_video_hosvd_core, print_video_hosvd_factors)[frame_index, :, :, :])
            plt.title(f'Reconstructed HOSVD image at compression rate = {print_rate_hosvd}')
            plt.show()
            plt.figure()
            plt.imshow(V[frame_index, :, :, :])
            plt.title(f'Original image')
            plt.show()

            # save files
            numpy2video('report4_ttscd.mp4', reconstruct(print_video_ttsvd))
            numpy2video('report4_hosvd.mp4', hosvd_recompose(print_video_hosvd_core, print_video_hosvd_factors))


    return None #modify as needed


def video2numpy(fname='project4.mp4'):
    """
    Convert mp4 video with filename fname into numpy array
    """
    import cv2
    cap = cv2.VideoCapture(fname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    A = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, A[fc] = cap.read()
        fc += 1

    cap.release()
    
    return A.astype(float)/255 #Scales A to contain values between 0 and 1

def numpy2video(output_fname, A, fps=30):
    """
    Convert numpy array A into mp4 video and save as output_fname
    fps: frames per second.
    """
    import cv2
    video_array = A*255 #assumes A contains values between 0 and 1
    video_array  = video_array.astype('uint8')
    height, width, _ = video_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fname, fourcc, fps, (width, height))

    for frame in video_array:
        out.write(frame)

    out.release()

    return None

#----------------------
if __name__=='__main__':
    #out = part2() Uncomment and modify as needed after completing part2
    part2(figures=True)
