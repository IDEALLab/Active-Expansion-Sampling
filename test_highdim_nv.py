"""
Experiments with Neighborhood-Voronoi sampling on high-dimensional test functions

Reference:
    Singh, P., Van Der Herten, J., Deschrijver, D., Couckuyt, I., & Dhaene, T. (2017). 
    A sequential sampling strategy for adaptive classification of computationally expensive data. 
    Structural and Multidisciplinary Optimization, 55(4), 1425-1438.
        
Author(s): Wei Chen (wchen459@umd.edu)
"""

import timeit
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
import pyDOE

from gpc import GPClassifier
from query_strategies import NV
from functions import two_spheres
        
    
if __name__ == "__main__":
#def high_nv(d):
    
    example = two_spheres
    get_label = example
    d = 7
    
    # Set boundaries for plots and test samples
    BD = np.ones((2,d))
    BD[0] *= -1.5
    BD[1] *= 1.5
    BD[1,0] += 3.0

    budget = 1000
    nb_init = 20
    nb_new = 10
    n_iter = (budget - nb_init)/nb_new + 1
    N = 500 # pool size

    ls = .5
    p = .0 # noise level
    
    # Generate test set
    D_test = np.random.uniform(BD[0,:], BD[1,:], (10**4,d))
    L_test = get_label(D_test, p=0)
    
    n_runs = 100
    f1ss = np.zeros(n_runs)
    times = np.zeros(n_runs)
    clf = GPClassifier(RBF(ls), optimizer=None)
    
    for j in range(n_runs):
    
        np.random.seed(j)
        
        # Generate initial samples using a Latin Hypercube design
        init_points = pyDOE.lhs(d, samples=nb_init)
        init_points = init_points * (BD[1]-BD[0]) + BD[0]
        new_points = init_points
    
        i = 0
        time = 0
        
        qs = NV(
             nb_new = nb_new,
             init_points = init_points,
             bounds = BD,
             pool_size = N
             )
                 
        for i in range(n_iter):
            
            print '***** Run: %d/%d Iteration: %d/%d *****' %(j+1, n_runs, i+1, n_iter)
            
            start_time = timeit.default_timer()
            
            new_labels = get_label(new_points, p)
            
            start_time = timeit.default_timer()
            new_points = qs.make_query(new_labels)
            end_time = timeit.default_timer()
            time += (end_time - start_time)
            
        clf.fit(qs.X, qs.L)
        try:
            L_pred = clf.predict(D_test)
        except IndexError:
            L_pred = -np.ones(D_test.shape[0])
        f1ss[j] = f1_score(L_test, L_pred)
        times[j] = time
        
    np.save('./results/%dd_p%d_nv_time' % (d, p*10), times)
    np.save('./results/%dd_p%d_nv_f1s' % (d, p*10), f1ss)
    
    f1s_mean = np.mean(f1ss)
    f1s_std = np.std(f1ss)
    print 'Average F1 score: %.2f +/- %.2f s' % (f1s_mean, 1.96*f1s_std/n_runs**.5)

    time_mean = np.mean(times)
    time_std = np.std(times)
    print 'Average running time: %.2f +/- %.2f s' % (time_mean, 1.96*time_std/n_runs**.5)
