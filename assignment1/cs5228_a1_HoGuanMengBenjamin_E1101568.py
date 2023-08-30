import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances



def clean(df_cars_dirty):
    """
    Handle all "dirty" records in the cars dataframe

    Inputs:
    - df_cars_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_cars_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_cleaned = df_cars_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################


    
    
    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_cleaned



def handle_nan(df_cars_nan):
    """
    Handle all nan values in the cars dataframe

    Inputs:
    - df_cars_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_cars_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_no_nan = df_cars_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    
    

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_no_nan



def extract_facts(df_cars_facts):
    """
    Extract the facts as required from the cars dataset

    Inputs:
    - df_card_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################

    # Toy example -- assume question: What is the total number of listings?
    print('#listings: {}'.format(len(df_cars_facts)))
    print()

    
    
    

    ### Your code ends here #################################################################
    #########################################################################################


    

def kmeans_init(X, k, c1=None, method='kmeans++'):
    
    """
    Calculate the initial centroids for performin K-Means

    Inputs:
    - X: A numpy array of shape (N, F) containing N data samples with F features
    - k: number of centroids/clusters
    - c1: First centroid as the index of the data point in X
    - method: string that specifies the methods to calculate centroids ('kmeans++' or 'maxdist')

    Returns:
    - centroid_indices: NumPy array containing k centroids, represented by the
      indices of the respective data points in X
    """   
    
    centroid_indices = []
    
    # If the index of the first centroid index c1 is not specified, pick it randomly
    if c1 is None:
        c1 = np.random.randint(0, X.shape[0])
        
    # Add selected centroid index to list
    centroid_indices.append(c1)        
    
        
    # Calculate and add c2, c3, ..., ck 
    while len(centroid_indices) < k:
        
        c = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        
        ## Remember to cover the 2 cases 'kmeans++' and 'maxdist'
        

        
        
        ### Your code ends here #################################################################
        #########################################################################################                
            
        centroid_indices.append(c)
    
    # Return list of k centroid indices as numpy array (e.g. [0, 1, 2] for K=3)
    return np.array(centroid_indices)
