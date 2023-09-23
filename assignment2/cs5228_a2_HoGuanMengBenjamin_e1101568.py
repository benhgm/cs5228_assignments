import numpy as np

from sklearn.metrics.pairwise import euclidean_distances




def get_noise_dbscan(X, eps=0.0, min_samples=0):
    
    core_point_indices, noise_point_indices = None, None
    
    #########################################################################################
    ### Your code starts here ###############################################################

    ### 2.1 a) Identify the indices of all core points
    point_to_point_distances = euclidean_distances(X, X)

    core_border_points = np.where(point_to_point_distances <= eps, 1, 0)
    samples = core_border_points.sum(axis=0)
    core_point_indices = np.argwhere(samples>=min_samples).reshape((-1)).tolist()
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    
    
    #########################################################################################
    ### Your code starts here ###############################################################
    
    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices
    core_points = X[core_point_indices]
    distances_to_core_points = euclidean_distances(X, core_points)
    noise_points = np.where(distances_to_core_points > eps, 1, 0)
    noise_point_samples = noise_points.sum(axis=1)
    noise_point_indices = np.argwhere(noise_point_samples==core_points.shape[0]).reshape((-1)).tolist()
    

    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return core_point_indices, noise_point_indices