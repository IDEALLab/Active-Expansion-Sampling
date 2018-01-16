"""
Experiments with Active Expansion Sampling on high-dimensional test functions

Author(s): Wei Chen (wchen459@umd.edu)
"""

import timeit
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score

from gpc import GPClassifier
from query_strategies import AES
from functions import two_spheres

    
if __name__ == "__main__":
#def high_aes(d):
    
    example = two_spheres
    get_label = example
    d = 7
    
    # Set boundaries for plots and test samples
    BD = np.ones((2,d))
    BD[0] *= -1.5
    BD[1] *= 1.5
    BD[1,0] += 3.0
            
    init_point = np.zeros((1,d)) # initial point
    ls = 0.5 # length scale
    n_iter = 1000 # number of iterations/queries
    N = 500 # pool size
    p = .0 # noise level
    
    margin = .3 # higher -> distance between points in exploration stage smaller; expand slower
    eta = 1.3 # affect the threshold t: higher -> more accurate
    
    # Generate test set
    D_test = np.random.uniform(BD[0,:], BD[1,:], (10**4,d))
    L_test = get_label(D_test, p=0)
    
    n_runs = 100
    f1ss = np.zeros(n_runs)
    times = np.zeros(n_runs)
    
    for j in range(n_runs):

        np.random.seed(j)
        
        qs = AES(
             init_point=init_point,
#             model=GPClassifier(RBF(1), n_restarts_optimizer=5),
             model=GPClassifier(RBF(ls), optimizer=None),
             margin=margin,
             eta=eta,
             pool_size=N
             )
        
        i = 0
        time = 0
                 
        for i in range(n_iter+1):
            
            print '***** Run: %d/%d Iteration: %d/%d *****' %(j+1, n_runs, i, n_iter)
            
            start_time = timeit.default_timer()
            new = qs.make_query()
            end_time = timeit.default_timer()
            time += (end_time - start_time)
            
            l = get_label(new, p)
            qs.L = np.append(qs.L, l)
            
        clf = qs.model
        L_pred = clf.predict(D_test)
        f1ss[j] = f1_score(L_test, L_pred)
        times[j] = time
        
    np.save('./results/%dd_p%d_aes_%d_%d_time' % (d, p*10, margin*10, eta*10), times)
    np.save('./results/%dd_p%d_aes_%d_%d_f1s' % (d, p*10, margin*10, eta*10), f1ss)
    
    f1s_mean = np.mean(f1ss)
    f1s_std = np.std(f1ss)
    print 'Average F1 score: %.2f +/- %.2f s' % (f1s_mean, 1.96*f1s_std/n_runs**.5)

    time_mean = np.mean(times)
    time_std = np.std(times)
    print 'Average running time: %.2f +/- %.2f s' % (time_mean, 1.96*time_std/n_runs**.5)
