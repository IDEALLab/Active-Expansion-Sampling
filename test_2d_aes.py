"""
Experiments with Active Expansion Sampling on 2D test functions

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

from gpc import GPClassifier
from query_strategies import AES
from functions import one_circle, two_circles, branin, hosaki, beam
        
    
if __name__ == "__main__":
    
    example = branin
    get_label = example
    
    # Set boundaries for plots and test samples
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
           
    # Set the initial point
    init_points = {'one_circle':    np.array([[0, 0]]),
                   'two_circles':   np.array([[0, 0]]),
                   'branin':        np.array([[3, 3]]),
                   'hosaki':        np.array([[3, 3]]),
                   'beam':          np.array([[0.05, 0.05]])
                   }
                  
    # Set the number of iterations
    n_iters = {'one_circle':    50,
               'two_circles':   300,
               'branin':        350,
               'hosaki':        200,
               'beam':          240
               }
    
    # Set the length scales forclaim Gaussian kernels
    length_scales = {'one_circle':    1.,
                     'two_circles':   .5,
                     'branin':        .9,
                     'hosaki':        .4,
                     'beam':          .005
                     }
    
    BD = BDs[example.__name__]
    init_point = init_points[example.__name__]
    ls = length_scales[example.__name__]

    n_iter = n_iters[example.__name__]#interval*im_rows*im_cols
    N = 500 # pool size
    p = .0 # Bernoulli noise level
    s = .0 # Gaussian noise level
    
    margin = .3 # higher -> distance between points in exploration stage smaller; expand slower
    eta = 1.3 # affect the threshold t: higher -> more accurate
    
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
                    
        qs = AES(
             init_point=init_point,
#             model=GPClassifier(RBF(ls), n_restarts_optimizer=5),
             model=GPClassifier(RBF(ls), optimizer=None),
             margin=margin,
             eta=eta,
             pool_size=N
             )
        
        i = 0
        time = 0
        f1s = np.zeros(n_iter+1)
                 
        for i in range(n_iter+1):
            
            print '***** Run: %d/%d Iteration: %d/%d *****' %(j+1, n_runs, i, n_iter)
            
            start_time = timeit.default_timer()
            new = qs.make_query()
            end_time = timeit.default_timer()
            time += (end_time - start_time)
            
            clf = qs.model
            
            if L is not None:
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
            else:
                f1s[i] = 0

            l = get_label(new, p, s)
            qs.L = np.append(qs.L, l)
            D = qs.D
            L = qs.L
    
            if i==n_iter and j==0:#i>0 and i%interval==0:
                
                print 'Plotting ...'
    
                # Arrange subplots
                plt.figure(figsize=(5, 5))
                
                ax = plt.subplot(111)
#                plt.scatter(new[0,0], new[0,1], s=130, c='y', marker='*')
                plt.scatter(D[:, 0][L==-1], D[:, 1][L==-1], s=40, c='k', marker='x')
                plt.scatter(D[:, 0][L==1], D[:, 1][L==1], s=30, c='k', edgecolor='none')
    
                if not qs.is_initial:
                
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
                
                from matplotlib import rcParams
                rcParams.update({'font.size': 16})
                if example == beam:
                    plt.xlabel('Breadth (m)')
                    plt.ylabel('Height (m)')
                else:
                    plt.xticks(())
                    plt.yticks(())
                    plt.tight_layout()
                plt.xlim(BD[:,0])
                plt.ylim(BD[:,1])
                plt.savefig('./results/%s_p%d_s%d_aes_%d_%d.svg' 
                            % (example.__name__, p*10, s*10, margin*10, eta*10))
                plt.close()
            
        times[j] = time
        f1ss[j] = f1s
        
    if not get_f1s_explored:
        np.save('./results/%s_p%d_s%d_aes_%d_%d_time' % (example.__name__, p*10, s*10, margin*10, eta*10), times)
        np.save('./results/%s_p%d_s%d_aes_%d_%d_f1s' % (example.__name__, p*10, s*10, margin*10, eta*10), f1ss)
    else:
        np.save('./results/%s_p%d_s%d_aes_%d_%d_time_explored' % (example.__name__, p*10, s*10, margin*10, eta*10), times)
        np.save('./results/%s_p%d_s%d_aes_%d_%d_f1s_explored' % (example.__name__, p*10, s*10, margin*10, eta*10), f1ss)

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
