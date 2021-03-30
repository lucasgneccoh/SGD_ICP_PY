#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    
    
    p = ref.mean(1).reshape(-1,1)
    p_prime = data.mean(1).reshape(-1,1)    
    U, S, V = np.linalg.svd((data-p_prime) @ (ref-p).T, full_matrices=False)
    V = V.T
    R = V @ U.T
    if np.linalg.det(R)<0:
        U[:,-1] *= -1
        R = V @ U.T
    
    T = p - R @ p_prime
    return R, T.reshape(-1,1)


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)
    
    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    tree = KDTree(ref.T)
    for i in range(max_iter):
        # Match points
        ind = tree.query(data_aligned.T, return_distance=False).squeeze()
        #ICP
        R, T = best_rigid_transform(data, ref[:,ind])
        data_aligned = R @ data + T
        neighbors_list.append(ind)        
        R_list.append(R)
        T_list.append(T)
        rms = RMS(data_aligned, ref[:,ind])
        RMS_list.append(rms)
        if rms < RMS_threshold: break
    return data_aligned, R_list, T_list, neighbors_list, RMS_list

def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit, resample=False):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)
    
    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []
    tree = KDTree(ref.T)
    data_sample_ind = np.random.choice(data.shape[1], size= min(data.shape[1],sampling_limit), replace=False)
    for i in range(max_iter):
        print(i)
        if resample:
            # Sample
            data_sample_ind = np.random.choice(data.shape[1], size= min(data.shape[1],sampling_limit), replace=False)
        
    
        # Match points
        ind = tree.query(data_aligned[:,data_sample_ind].T, return_distance=False).squeeze()
        #ICP
        R, T = best_rigid_transform(data[:,data_sample_ind], ref[:,ind])
        
        data_aligned = R @ data + T
        
        neighbors_list.append((data_sample_ind,ind))        
        R_list.append(R)
        T_list.append(T)
        ind = tree.query(data_aligned.T, return_distance=False).squeeze()
        rms = RMS(data_aligned, ref[:,ind])
        RMS_list.append(rms)
        if rms < RMS_threshold: break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list



def RMS(c1, c2):
    return np.sqrt(np.mean(np.sum(np.power(c1 - c2, 2), axis=0)))

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:
        print("*************** 2 ICP simple ******************")
        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R @ bunny_r + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        RMS_before = RMS(bunny_r, bunny_o)

        RMS_after = RMS(bunny_r_opt, bunny_o)

        print('Average RMS between points :')
        print('Before = {:.6f}'.format(RMS_before))
        print(' After = {:.6f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:
        print("*************** 3 ICP 2D ******************")
        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        
        plt.figure()
        plt.plot(RMS_list)
        plt.title("RMS for the 2D example")
        plt.xlabel("iteration")
        plt.ylabel("RMS value")
        plt.show()
        

    # If statement to skip this part if wanted
    if False:
        print("*************** 3 ICP 3D ******************")
        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        prop = 0.3
        l_p, l_o = bunny_p.shape[1], bunny_o.shape[1]
        sample_p = bunny_p[:,np.random.choice(l_p, int(prop*l_p), replace=False)] 
        sample_o = bunny_o[:,np.random.choice(l_o, int(prop*l_o), replace=False)]
        show_ICP(sample_p, sample_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.figure()
        plt.plot(RMS_list)
        plt.title("RMS for the 3D bunny example")
        plt.xlabel("iteration")
        plt.ylabel("RMS value")
        plt.show()
        
        
    ''' Bonus '''
    # If statement to skip this part if wanted
    if True:
        sample_size = 1000
        resample = True
        # Cloud paths
        nd_ref_path = '../data/Notre_Dame_Des_Champs_1.ply'
        nd_data_path = '../data/Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        nd_ref_ply = read_ply(nd_ref_path)
        nd_data_ply = read_ply(nd_data_path)
        nd_ref = np.vstack((nd_ref_ply['x'], nd_ref_ply['y'], nd_ref_ply['z']))
        nd_data = np.vstack((nd_data_ply['x'], nd_data_ply['y'], nd_data_ply['z']))

        # Apply ICP
        nd_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(nd_data, nd_ref, 15, 1e-4, sample_size, resample=resample)
        
        # Show ICP
        # show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Apply the tranformation
        nd_opt = R_list[-1] @ nd_data + T_list[-1]

        # Save cloud
        write_ply('../nd_opt', [nd_opt.T], ['x', 'y', 'z'])
        
        
        # Plot RMS
        plt.figure()
        plt.plot(RMS_list)
        plt.title("RMS for the Notre Dame (with sampling of {} points)\nResample={}".format(sample_size, resample))
        plt.xlabel("iteration")
        plt.ylabel("RMS value")
        plt.show()
        
        #neighbor_change = []
        #for i in range(1,len(neighbors_list)):
        #    neighbor_change.append(np.nonzero(np.array(neighbors_list[i])-np.array(neighbors_list[i-1]))[0].sum())
       # 
       # plt.figure()
       # plt.plot(neighbor_change)
       # plt.ylabel('number of match changes')
       # plt.xlabel('iteration')
       # plt.show()
        
            

