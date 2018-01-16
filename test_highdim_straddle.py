"""
Experiments with the Straddle heuristic on high-dimensional test functions

Reference:
    Bryan, Brent, et al. "Active learning for identifying function threshold boundaries." 
    Advances in neural information processing systems. 2006.
        
Author(s): Wei Chen (wchen459@umd.edu)
"""

import timeit
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
from libact.base.dataset import Dataset

from gpc import GPClassifier
from query_strategies import Straddle
from functions import two_spheres
        
    
if __name__ == "__main__":
    
    example = two_spheres
    get_label = example
    d = 7
    
    # Set boundaries for plots and test samples
    BD = np.ones((2,d))
    BD[0] *= -1.5
    BD[1] *= 1.5
    BD[1,0] += 3.0
               
    ls = .5 # length scale
    n_iter = 1000 # number of iterations/queries
    N = 1000 # pool size
    p = .0 # noise level
    
    # Generate test set
    D_test = np.random.uniform(BD[0,:], BD[1,:], (10**4,d))
    L_test = get_label(D_test, p=0)
    
    n_runs = 100
    f1ss = np.zeros(n_runs)
    times = np.zeros(n_runs)
    
    for j in range(n_runs):
    
        np.random.seed(j)
        
        D = None
        L = None
        qs = None
        is_initial = True
        pool = np.random.uniform(BD[0], BD[1], (N,d))
        labels = [None]*pool.shape[0]
        dataset = Dataset(pool,labels)
    
        i = 0
        time = 0
                 
        for i in range(n_iter+1):
            
            print '***** Run: %d/%d Iteration: %d/%d *****' %(j+1, n_runs, i, n_iter)
            
            start_time = timeit.default_timer()
            
            if np.any(L==-1) and np.any(L==1):
                print 'Selecting a sample to query ...'
                is_initial = False
                
                if qs is None:
                    
                    qs = Straddle(
                         dataset, # Dataset object
                         model=GPClassifier(RBF(ls), optimizer=None)
                         )
                    
                ask_id = qs.make_query()
                clf = qs.model
                new = pool[ask_id].reshape(1,-1)
                
            else:
                print 'Selecting a sample to query by diverse sampling ...'
                if i == 0:
                    idx = np.random.choice(pool.shape[0])
                else:
                    # Maximize the minimum distance between the initial pool and labeled data
                    distances = pairwise_distances(pool, D)
                    scores = np.min(distances, axis=1)
                    idx = np.argmax(scores)
                new = pool[idx].reshape(1,-1)
                ask_id = np.arange(pool.shape[0])[idx]
    
            end_time = timeit.default_timer()
            time += (end_time - start_time)
            
            l = get_label(new, p)
            dataset.update(ask_id, l) # update dataset
            
            if i == 0:
                D = new
                L = [l]
            else:
                D = np.append(D, new, axis=0)
                L = np.append(L, l)
            
        clf = qs.model
        L_pred = clf.predict(D_test)
        f1ss[j] = f1_score(L_test, L_pred)
        times[j] = time
        
    np.save('./results/%dd_p%d_straddle_time' % (d, p*10), times)
    np.save('./results/%dd_p%d_straddle_f1s' % (d, p*10), f1ss)
    
    f1s_mean = np.mean(f1ss)
    f1s_std = np.std(f1ss)
    print 'Average F1 score: %.2f +/- %.2f s' % (f1s_mean, 1.96*f1s_std/n_runs**.5)

    time_mean = np.mean(times)
    time_std = np.std(times)
    print 'Average running time: %.2f +/- %.2f s' % (time_mean, 1.96*time_std/n_runs**.5)
