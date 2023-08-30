import numpy as np
import matplotlib.pyplot as plt


def plot_centroid_distributions(func, X, k, c1, num_runs=100, method='maxdist'):

    cdict = {}
    for i in range(1, k+1):
        cdict[i] = {}
    
    cdict[1] = {c1: num_runs}
    
    for i in range(num_runs):
        centroid_indices = func(X, k, c1=c1, method=method)

        for idx, c in enumerate(centroid_indices):
            if idx == 0:
                continue
            if c in cdict[idx+1]:
                cdict[idx+1][c] += 1
            else:
                cdict[idx+1][c] = 1
    
    
    for i in range(1, k+1):
        distribution = cdict[i]
        keys = distribution.keys()
        values = distribution.values()
        plt.figure()
        #plt.title('Distribution of data points for c{}'.format(i))
        plt.bar(keys, values)
        plt.xlim((-1, 15))
        plt.xticks(fontsize=18)
        plt.tick_params(top=False, bottom=True, left=False, right=False, labelleft=False, labelbottom=True)
        plt.tight_layout()
        plt.show()    
