"""
Experiments with the Straddle heuristic on 2D test functions

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as patches
from libact.base.dataset import Dataset

from gpc import GPClassifier
from query_strategies import Straddle
from functions import one_circle, two_circles, branin, hosaki, beam

    
if __name__ == "__main__":
    
    example = branin
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
                'beam_tight':             np.array([[0, 0.02],
                                           [0.1, 0.16]])
                }
                  
    # Set the number of iterations
    n_iters = {'one_circle':    50,
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
    ls = length_scales[example.__name__]

    n_iter = n_iters[example.__name__]
    N = 500 # pool size
    p = .2 # Bernoulli noise level
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
    
    for j in range(n_runs):
    
        np.random.seed(j)
        
        D = None
        L = None
        qs = None
        is_initial = True
        pool_BD = pool_BDs[example.__name__+'_'+bound_size] # change the size of variable bounds
        pool = np.random.uniform(pool_BD[0], pool_BD[1], (N,2))
        labels = [None]*pool.shape[0]
        dataset = Dataset(pool,labels)
    
        i = 0
        time = 0
        f1s = np.zeros(n_iter+1)
                 
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
                new = pool[ask_id].reshape(1,-1)
                clf = qs.model
                
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
                
            if i == 0:
                f1s[i] = 0
            else:
                if np.all(L==1):
                    L_pred = np.ones_like(L_test)
                    f1s[i] = f1_score(L_test, L_pred)
                elif np.all(L==-1):
                    L_pred = -1*np.ones_like(L_test)
                    f1s[i] = f1_score(L_test, L_pred)
                else:
                    # Compute the test accuracy
                    if get_f1s_explored:
                        mean_test, var_test = clf.predict_mean_var(D_test)
                        ind = (np.abs(mean_test)+margin)/np.sqrt(var_test) > margin*eta
                        L_pred = clf.predict(D_test[ind])
                        f1s[i] = f1_score(L_test[ind], L_pred)
                    else:
                        L_pred = clf.predict(D_test)
                        f1s[i] = f1_score(L_test, L_pred)
                        
            l = get_label(new, p, s)
            dataset.update(ask_id, l) # update dataset
            
            if i == 0:
                D = new
                L = l
            else:
                D = np.append(D, new, axis=0)
                L = np.append(L, l)
    
            if i==n_iter and j==0:#i>0 and i%interval==0:
                
                print 'Plotting ...'
                
                # Arrange subplots
                plt.figure(figsize=(5, 5))
                
                ax = plt.subplot(111)
#                plt.scatter(new[0,0], new[0,1], s=130, c='y', marker='*')
                plt.scatter(D[:, 0][L==-1], D[:, 1][L==-1], s=40, c='k', marker='x')
                plt.scatter(D[:, 0][L==1], D[:, 1][L==1], s=30, c='k', edgecolor='none')
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
    
                if not is_initial:
                
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
                plt.savefig('./results/%s_p%d_s%d_straddle_%s.svg' % (example.__name__, p*10, s*10, bound_size))
                plt.close()
            
        times[j] = time
        f1ss[j] = f1s
    
    if not get_f1s_explored:
        np.save('./results/%s_p%d_s%d_straddle_%s_time' % (example.__name__, p*10, s*10, bound_size), times)
        np.save('./results/%s_p%d_s%d_straddle_%s_f1s' % (example.__name__, p*10, s*10, bound_size), f1ss)
    else:
        np.save('./results/%s_p%d_s%d_straddle_%s_time_explored' % (example.__name__, p*10, s*10, bound_size), times)
        np.save('./results/%s_p%d_s%d_straddle_%s_f1s_explored' % (example.__name__, p*10, s*10, bound_size), f1ss)
    
    time_mean = np.mean(times)
    time_std = np.std(times)
    print 'Average running time: %.2f +/- %.2f s' % (time_mean, 1.96*time_std/n_runs**.5)
    
    f1s_mean = np.mean(f1ss, axis=0)
    plt.figure()
    plt.plot(f1s_mean)
    plt.ylim(0.0,1.0)
    plt.xlabel('Number of queries')
    plt.ylabel('F1 score')
    plt.show()
