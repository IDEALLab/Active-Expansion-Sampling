"""
Experiments with Neighborhood-Voronoi sampling on 2D test functions

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as patches
import pyDOE

from gpc import GPClassifier
from query_strategies import NV
from functions import one_circle, two_circles, branin, hosaki, beam

    
if __name__ == "__main__":
    
    example = hosaki
    get_label = example
    bound_size = 'tight'
    
    # Set boundaries for plots
    BDs = {'one_circle':    np.array([[-8, -7], 
                                      [8, 9]]),
           'two_circles':   np.array([[-2, -2], 
                                      [5, 2]]),
           'branin':        np.array([[-13, -8], 
                                      [18, 23]]),
           'hosaki':        np.array([[-3, -3.5], 
                                      [9, 8.5]]),
           'beam':          np.array([[0, 0.02],
                                      [0.1, 0.16]])
           }
           
    pool_BDs = {'one_circle_tight':       np.array([[-8, -7], 
                                           [8, 9]]),
                'two_circles_tight':      np.array([[-1.5, -1.5], 
                                           [4.5, 1.5]]),
                'branin_tight':      np.array([[-6,-1], 
                                           [12, 17]]), # tight
                'branin_insuff':    np.array([[-4, -2], 
                                           [11, 14]]), # insufficient
                'branin_loose':      np.array([[-13, -8], 
                                           [18, 23]]), # loose
                'hosaki_tight':      np.array([[0, 0], 
                                           [6, 5]]), # tight
                'hosaki_insuff':    np.array([[1, 0], 
                                           [6, 4.5]]), # insufficient
                'hosaki_loose':      np.array([[-2.5, -3], 
                                           [8.5, 8]]), # loose
                'beam_tight':        np.array([[0, 0.02],
                                           [0.1, 0.16]])
                }
                  
    # Set the number of iterations
    budgets = {'one_circle':    50,
               'two_circles':   300,
               'branin':        350,
               'hosaki':        200,
               'beam':          240
               }
    
    # Set the length scales for Gaussian kernels
    length_scales = {'one_circle':    1.,
                     'two_circles':   .5,
                     'branin':        .9,
                     'hosaki':        .4,
                     'beam':          .005
                     }
    
    BD = BDs[example.__name__]
    pool_BD = pool_BDs[example.__name__+'_'+bound_size] # proper variable bounds
    ls = length_scales[example.__name__]

    budget = budgets[example.__name__]
    
    nb_init = 10
    nb_new = 10
    n_iter = (budget - nb_init)/nb_new + 1
    N = 500 # pool size
    p = .0 # Bernoulli noise level
    s = .0 # Gaussian noise level
    
    get_f1s_explored = False
    
    # Generate test set
    xx, yy = np.meshgrid(np.linspace(BD[0][0], BD[1][0], 100),
                         np.linspace(BD[0][1], BD[1][1], 100))
    D_test = np.vstack((xx.ravel(), yy.ravel())).T
    L_test = get_label(D_test)
    
    n_runs = 100
    f1ss = np.zeros((n_runs, n_iter+1))
    times = np.zeros(n_runs)
    clf = GPClassifier(RBF(ls), optimizer=None)
    
    for j in range(n_runs):
    
        np.random.seed(j)
        
        # Generate initial samples using a Latin Hypercube design
        init_points = pyDOE.lhs(2, samples=nb_init)
        init_points = init_points * (pool_BD[1]-pool_BD[0]) + pool_BD[0]
        new_points = init_points
        
        i = 0
        time = 0
        f1s = np.zeros(n_iter+1)
        
        qs = NV(
             nb_new = nb_new,
             init_points = init_points,
             bounds = pool_BD,
             pool_size = N
             )
                 
        for i in range(n_iter):
            
            print '***** Run: %d/%d Iterations: %d/%d *****' %(j+1, n_runs, i+1, n_iter)
            
            new_labels = get_label(new_points, p, s)
            
            start_time = timeit.default_timer()
            new_points = qs.make_query(new_labels)
            end_time = timeit.default_timer()
            time += (end_time - start_time)
            
            D = qs.X
            L = qs.L
            clf.fit(D, L)
            print D.shape[0]
    
            if j==0 and i==n_iter-1:
                
                print 'Plotting ...'
                
                # Arrange subplots
                plt.figure(figsize=(5, 5))
                
                ax = plt.subplot(111)
#                plt.scatter(new[0,0], new[0,1], s=130, c='y', marker='*')
                plt.scatter(D[:, 0][L==-1], D[:, 1][L==-1], s=40, c='k', marker='x')
                plt.scatter(D[:, 0][L==1], D[:, 1][L==1], s=30, c='k', edgecolor='none')
                
#                if i > 1:
#                    idx = 20
#                    plt.scatter(D[idx, 0], D[idx, 1], s=40, c='r', marker='+', edgecolor='none')
#                    plt.scatter(D[qs.Nbh[idx]][:, 0], D[qs.Nbh[idx]][:, 1], s=30, c='r', edgecolor='none')
                
                ax.add_patch(
                    patches.Rectangle(
                        pool_BD[0], # (x,y)
                        pool_BD[1,0]-pool_BD[0,0], # width
                        pool_BD[1,1]-pool_BD[0,1], # height
                        fill=False,
                        linestyle='dashed',
                        linewidth=2,
                        color='g'
                    )
                )
                
                # Create a mesh grid
                xx, yy = np.meshgrid(np.linspace(BD[0][0], BD[1][0], 200),
                                     np.linspace(BD[0][1], BD[1][1], 200))
                
                # Plot the decision function for each datapoint on the grid
                grid = np.vstack((xx.ravel(), yy.ravel())).T
                Z0 = clf.predict_real(grid)[:,-1] # to show probability
                Z0 = Z0.reshape(xx.shape)
                Z1, Z4 = clf.predict_mean_var(grid) # to show posterior mean and variance
                Z1 = Z1.reshape(xx.shape)
                Z4 = Z4.reshape(xx.shape)
                Z3 = get_label(grid) # to show ground truth decision boundary
                Z3 = Z3.reshape(xx.shape)
                
                plt.axis('equal')
                plt.contour(xx, yy, Z1, levels=[0], linewidths=3, alpha=0.5) # estimated decision boundary
                image = plt.imshow(Z3<0, interpolation='nearest',
                                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                                   aspect='auto', origin='lower', 
                                   cmap=plt.get_cmap('gray'), alpha=.5) # ground truth domain
                
                plt.xticks(())
                plt.yticks(())
                plt.tight_layout()
                plt.xlim(BD[:,0])
                plt.ylim(BD[:,1])
                plt.savefig('./results/%s_p%d_s%d_nv_%s.svg' % (example.__name__, p*10, s*10, bound_size))
                plt.close()
            
            # Compute the test accuracy
            if np.all(L==1):
                L_pred = np.ones_like(L_test)
                f1s[i+1] = f1_score(L_test, L_pred)
            elif np.all(L==-1):
                L_pred = -1*np.ones_like(L_test)
                f1s[i+1] = f1_score(L_test, L_pred)
            else:
                if get_f1s_explored:
                    mean_test, var_test = clf.predict_mean_var(D_test)
                    ind = (np.abs(mean_test)+margin)/np.sqrt(var_test) > margin*eta
                    L_pred = clf.predict(D_test[ind])
                    f1s[i+1] = f1_score(L_test[ind], L_pred)
                else:
                    L_pred = clf.predict(D_test)
                    f1s[i+1] = f1_score(L_test, L_pred)
            
        times[j] = time
        f1ss[j] = f1s
    
    if not get_f1s_explored:
        np.save('./results/%s_p%d_s%d_nv_%s_time' % (example.__name__, p*10, s*10, bound_size), times)
        np.save('./results/%s_p%d_s%d_nv_%s_f1s' % (example.__name__, p*10, s*10, bound_size), f1ss)
    else:
        np.save('./results/%s_p%d_s%d_nv_%s_time_explored' % (example.__name__, p*10, s*10, bound_size), times)
        np.save('./results/%s_p%d_s%d_nv_%s_f1s_explored' % (example.__name__, p*10, s*10, bound_size), f1ss)
    
    time_mean = np.mean(times)
    time_std = np.std(times)
    print 'Average running time: %.2f +/- %.2f s' % (time_mean, 1.96*time_std/n_runs**.5)
    
    f1s_mean = np.mean(f1ss, axis=0)
    sample_sizes = np.zeros(n_iter+1)
    sample_sizes[1:] = np.arange(nb_init, budget+1, nb_new)
    plt.figure()
    plt.plot(sample_sizes, f1s_mean)
    plt.ylim(0.0,1.0)
    plt.xlabel('Number of queries')
    plt.ylabel('F1 score')
    plt.show()
