import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import display


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

    ##############################################
    ### Step 1: Remove Duplicated Rows of Data ###
    ##############################################
    df_cars_cleaned = df_cars_cleaned.drop_duplicates()
    df_cars_cleaned.reset_index(drop=True, inplace=True)
    assert len(df_cars_cleaned) == 14344, "expected 14,344 entries after duplicates are cleared"
    # display(df_cars_cleaned)

    ###########################################################################
    ### Step 2: Correct Data Errors in Rows with 'mercedes' as Vehicle Make ###
    ###########################################################################
    df_cars_mercedes_error = df_cars_cleaned.loc[df_cars_cleaned["make"] == "mercedes"]

    df_cars_a200 = df_cars_mercedes_error.loc[(df_cars_mercedes_error["manufactured"]==2018) & (df_cars_mercedes_error["power"]==120) & (df_cars_mercedes_error["curb_weight"]=="1375")]
    df_cars_a180 = df_cars_mercedes_error.loc[(df_cars_mercedes_error["manufactured"]==2020) & (df_cars_mercedes_error["power"]== 96) & (df_cars_mercedes_error["curb_weight"]=="1365")]
    
    df_cars_a200["model"] = df_cars_a200["model"].apply(lambda x: "a200")
    df_cars_a180["model"] = df_cars_a180["model"].apply(lambda x: "a180")
   
    for idx in df_cars_a200.index:
        df_cars_cleaned.loc[idx, "make"] = "mercedes-benz"
        df_cars_cleaned.loc[idx, "model"] = "a200"
        display(df_cars_cleaned.loc[[idx]])
    
    for idx in df_cars_a180.index:
        df_cars_cleaned.loc[idx, "make"] = "mercedes-benz"
        df_cars_cleaned.loc[idx, "model"] = "a180"
        display(df_cars_cleaned.loc[[idx]])

    ##################################################
    ### Step 3a: Clean up "curb_weight" Dirty Data ###
    ##################################################
    df_cars_xxxx = df_cars_cleaned.loc[df_cars_cleaned["curb_weight"] == "XXXXX"]

    for idx, row in df_cars_xxxx.iterrows():
        correct_entry = df_cars_cleaned.loc[(df_cars_cleaned["manufactured"] == row["manufactured"]) 
                                            & (df_cars_cleaned["power"] == row["power"])
                                            & (df_cars_cleaned["engine_cap"] == row["engine_cap"])
                                            & (df_cars_cleaned["make"] == row["make"])
                                            & (df_cars_cleaned["model"] == row["model"])
                                            & (df_cars_cleaned["type_of_vehicle"] == row["type_of_vehicle"])]
        correct_curb_weight = correct_entry["curb_weight"].unique()
        correct_curb_weight = correct_curb_weight.tolist()
        correct_curb_weight.remove("XXXXX")

        if len(correct_curb_weight) != 1:
            continue
        else:
            df_cars_xxxx.loc[idx, "curb_weight"] = correct_curb_weight[0]
    
    for idx in df_cars_xxxx.index:
        df_cars_cleaned.loc[idx, "curb_weight"] = df_cars_xxxx.loc[idx, "curb_weight"]
    
    for col in df_cars_cleaned.columns:
        print(col, df_cars_cleaned[col].unique())

    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned.curb_weight != "XXXXX"]

    #####################################################
    ### Step 3b: Convert "curb_weight" data to integer ###
    #####################################################
    df_cars_cleaned["curb_weight"] = df_cars_cleaned["curb_weight"].astype(int)
    display(df_cars_cleaned.dtypes)
    
    
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

    ####################################################
    ### Step 1: Fill in missing data in "url" column ###
    ####################################################
    df_cars_no_nan["url"] = df_cars_no_nan["listing_id"].apply(lambda x: "https://www.sgcarmart.com/listing/" + str(x))
    # display(df_cars_no_nan.isna().sum())
    # display(df_cars_no_nan)

    #####################################################
    ### Step 2: Fill in missing data in "make" column ###
    #####################################################
    df_cars_nan_make = df_cars_no_nan.loc[df_cars_no_nan["make"].isna()]
    # display(df_cars_nan_make)
    for idx, row in df_cars_nan_make.iterrows():
        correct_make = df_cars_no_nan.loc[(df_cars_no_nan["model"] == row["model"]) & (df_cars_no_nan["make"].notnull())]
        correct_make = correct_make["make"].unique()
        correct_make = correct_make.tolist()[0]
        
        df_cars_no_nan.loc[idx, "make"] = correct_make
    # display(df_cars_no_nan.isna().sum())
    # display(df_cars_no_nan)

    ########################################################
    ### Step 3: Fill in missing data in "mileage" column ###
    ########################################################
    df_cars_nan_mileage = df_cars_no_nan.loc[df_cars_no_nan["mileage"].isna()]
    # display(df_cars_nan_mileage)
    df_cars_no_nan["mileage"].fillna(df_cars_no_nan.groupby(['no_of_owners'])['mileage'].transform('median'), inplace=True)
    # display(df_cars_no_nan.loc[df_cars_nan_mileage.index])

    ########################################################
    ### Step 3: Fill in missing data in "price" column ###
    ########################################################
    df_cars_nan_price = df_cars_no_nan.loc[df_cars_no_nan["price"].isna()]
    # display(df_cars_nan_price)
    df_cars_no_nan["price"].fillna(df_cars_no_nan.groupby(['no_of_owners', 'type_of_vehicle'])['price'].transform('median'), inplace=True)
    # display(df_cars_no_nan.loc[df_cars_nan_price.index])
    

    
    

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
    print(f"1) highest price: {df_cars_facts['price'].max()}\n   lowest price: {df_cars_facts['price'].min()}")
    print(f"2) number of different car makes: {df_cars_facts['make'].nunique()}")
    print(f"3) number of sold toyota corollas manufactured before 2010: {len(df_cars_facts.loc[(df_cars_facts['model']=='corolla') & (df_cars_facts['manufactured'] < 2010)])}")
    
    makes = [[x, len(df_cars_facts.loc[df_cars_facts['make']==x])] for x in df_cars_facts['make'].unique()]
    sorted_makes = sorted(makes, key=lambda x: x[1], reverse=True)
    print(f"4) top-3 most sold cars and their corresponding sales:\n\ti. {sorted_makes[0][0]} - {sorted_makes[0][1]}\n\tii. {sorted_makes[1][0]} - {sorted_makes[1][1]}\n\tiii. {sorted_makes[2][0]} - {sorted_makes[2][1]}")
    
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
