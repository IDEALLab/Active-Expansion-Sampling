"""
Query strategies for active learning (adaptive sampling)

Author(s): Wei Chen (wchen459@umd.edu)
"""


from itertools import combinations
import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, zip

    
def gen_candidates(N, c, r):
     
    c = c.flatten()
    pool = []
    i = 0
    
    while i < N:
        sample = np.random.uniform(c-r, c+r)
        # Reject samples
        if np.linalg.norm(sample-c) <= r:
            pool.append(sample)
            i += 1
    
    candidates = np.array(pool)
    
    return candidates


class Straddle(QueryStrategy):

    """ 
    Straddle Heuristic 
    Reference:
        Bryan, Brent, et al. "Active learning for identifying function threshold boundaries." 
        Advances in neural information processing systems. 2006.
    """
    
    def __init__(self, *args, **kwargs):
        super(Straddle, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
    
    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        self.model.fit(*(dataset.format_sklearn()))

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        mean, var = self.model.predict_mean_var(X_pool)
        score = 1.96*var**.5 - np.abs(mean)
        ask_id = np.argmax(score)

        return unlabeled_entry_ids[ask_id]


class NV(object):

    """ 
    Neighborhood-Voronoi sequential sampling 
    Reference:
        Singh, P., Van Der Herten, J., Deschrijver, D., Couckuyt, I., & Dhaene, T. (2017). 
        A sequential sampling strategy for adaptive classification of computationally expensive data. 
        Structural and Multidisciplinary Optimization, 55(4), 1425-1438.
    """
    
    def __init__(self, *args, **kwargs):
        self.nb_new = kwargs.pop('nb_new', 10)
        self.init_points = kwargs.pop('init_points', np.random.rand(15, 2))
        self.dim = self.init_points.shape[1]
        self.X = None
        self.L = None
        self.X_delta = self.init_points
        self.m = 2*self.dim
        self.Nbh = np.zeros((self.init_points.shape[0], self.m), dtype=int)
        self.nbh_scores = np.zeros(self.init_points.shape[0])
        self.bounds = kwargs.pop('bounds', np.array([[0,0],[1,1]]))
        self.pool_size = kwargs.pop('pool_size', 50**self.dim)
        
    def eval_neighborhood(self, pr_id, nbr_id):
        ''' Get neighborhood scores for a reference point X[pr_id] '''
        Nbh = self.X[nbr_id]
        coh = np.mean(np.linalg.norm(Nbh-self.X[pr_id], axis=1)) # cohesion
        p_dist = pairwise_distances(Nbh)
        np.fill_diagonal(p_dist, np.inf)
        p_dist_min = np.min(p_dist, axis=1)
        adh = np.mean(p_dist_min) # adhesion
        r = adh/np.sqrt(2)/coh # cross-polytope ratio (dim > 1)
        nbh_score = r/coh # neighborhood score
        return nbh_score
        
    def eval_nbhs(self, pr_id, nbr_ids):
        ''' Given candidate neigbhorhoods of X[pr_id], get the neighberhood with the maximum score '''
        for nbr_id in nbr_ids:
            nbr_id = np.array(nbr_id)
            nbh_score = self.eval_neighborhood(pr_id, nbr_id)
            if nbh_score > self.nbh_scores[pr_id]:
                self.nbh_scores[pr_id] = nbh_score
                self.Nbh[pr_id] = nbr_id
    
    def make_query(self, L_delta):
        if self.X is None:
            # Constructing neighborhoods for the initial iteration
            # Run an exhausted search to find best neighborhood for each initial points
            self.X = self.X_delta
            self.L = L_delta
            for pr_id in range(self.X.shape[0]):
                s = range(self.X.shape[0])
                s.remove(pr_id)
                nbr_ids = set(combinations(s, self.m))
                self.eval_nbhs(pr_id, nbr_ids)
                        
        else:
            n = self.X.shape[0]
            self.X = np.vstack((self.X, self.X_delta))
            self.L = np.hstack((self.L, L_delta))
            # Randomly initialize the neighborhoods of X_delta
            Nbh_delta = np.zeros((self.nb_new, self.m))
            for i in range(self.nb_new):
                Nbh_delta[i] = np.random.choice(n, size=self.m, replace=False)
            self.Nbh = np.vstack((self.Nbh, Nbh_delta))
            self.nbh_scores = np.hstack((self.nbh_scores, np.zeros(self.nb_new)))
            for i in range(n, n+self.nb_new):
                for j in range(n):
                    # Try to add X_delta[i] to neighborhood of X[j]
                    nbr_ids = np.repeat(self.Nbh[j].reshape((1,-1)), self.m, axis=0)
                    np.fill_diagonal(nbr_ids, i)
                    nbr_ids = np.array(nbr_ids, dtype=int)
                    self.eval_nbhs(j, nbr_ids)
                    # Try to add X[j] to neighborhood of X[i]
                    nbr_ids = np.repeat(self.Nbh[i].reshape((1,-1)), self.m, axis=0)
                    np.fill_diagonal(nbr_ids, j)
                    nbr_ids = np.array(nbr_ids, dtype=int)
                    self.eval_nbhs(i, nbr_ids)
                n += 1
                
        # Calculate class disagreement
        n = self.X.shape[0]
        class_disagr = np.zeros(n)
        self.Nbh = np.array(self.Nbh, dtype=int)
        for i in range(n):
            if max(self.L[self.Nbh[i]]) != min(self.L[self.Nbh[i]]):
                class_disagr[i] = 1.0
            
        # Calculate Voronoi cell sizes
        L = self.pool_size
        T = np.random.uniform(self.bounds[0], self.bounds[1], (L, self.dim))
        A = np.zeros(L)
        vol = np.zeros(n)
        for k in range(L):
            closest_dist = np.inf
            for i in range(n):
                dist = np.linalg.norm(self.X[i]-T[k])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = i
            vol[closest_id] += 1.0/L
            A[k] = closest_id # assign each point in T to a point in X

        # Sort X according to combined scores of class disagreement + Voronoi cell size
        # and find highest ranked neighborhoods
        score = class_disagr + vol
        inds = score.argsort()[-self.nb_new:]

        # Select new samples from T
        self.X_delta = np.zeros((self.nb_new, self.dim))
        i = 0
        for idx in inds:
#            nbrs = np.append(self.Nbh[idx], idx) # indices of X[idx] and its neighbors
            others = np.append(self.X, self.X_delta[:i], axis=0) # other existing samples
            max_min_dist = 0
            for t in T[A==idx]:
                dist = np.linalg.norm(others-t, axis=1)
                min_dist = np.min(dist)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    self.X_delta[i] = t
            i += 1
            
        return self.X_delta


class AES(object):
    
    """ Active Expansion Sampling """
    
    def __init__(self, *args, **kwargs):
        
        self.init_point = kwargs.pop('init_point', 0)
        self.model = kwargs.pop('model', None)
        self.margin = kwargs.pop('margin', .7)
        self.eta = kwargs.pop('eta', 1.2)
        self.pool_size = kwargs.pop('pool_size', 500)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if self.init_point is None:
            self.init_point = np.random.rand(2)
        self.D = self.init_point
        self.L = np.array([],dtype=int)
        self.last = self.init_point
        self.is_initial = True
        
    def make_query(self):
        
        if len(self.L) == 0:
            print '1st sample'
            return self.init_point
            
        self.model.fit(self.D, self.L)
        print self.model.kernel_
        
        # Compute the radius of the local pool -- delta_explore and delta_exploit 
        # -- based on t, epsilon and length scale
        ls = self.model.kernel_.length_scale
        mu, nu = self.model.get_mu_nu()
        t = self.eta*self.margin
        
        if 1-nu < 1/self.eta**2: # k_m <= 1, delta_exploit has a feasible solution
            gamma = (math.log(self.eta**2*nu/(self.eta**2-1)))**.5
            self.delta_exploit = ls*gamma
            # Suppose exploitation stage, get candidate samples
            candidates = gen_candidates(self.pool_size, self.last, self.delta_exploit)
            # Compute probability of being in the other class with some certainty
            mean, var = self.model.predict_mean_var(candidates)
            prob = (np.abs(mean)+self.margin)/np.sqrt(var)
        else:
            self.delta_exploit = ls
            # In the case where delta_exploit does not have a feasible solution, 
            # we need to manually set its value such that the query strategy has 
            # feasible solutions in the local pool
            # (i.e., there are candidates with p_epsilon > tau)
            while True:
                candidates = gen_candidates(self.pool_size, self.last, self.delta_exploit)
                mean, var = self.model.predict_mean_var(candidates)
                prob = (np.abs(mean)+self.margin)/np.sqrt(var)
                
                # Check if there are feasible solutions
                if sum(prob<=t) > 0:
                    break
                else:
                    self.delta_exploit *= 1.5
        
        if self.is_initial:
            # Check if there are labels from both classes
            if np.any(self.L==-1) and np.any(self.L==1):
                self.is_initial = False
                self.is_exploiting = True
                self.center = np.mean(self.D[self.L==1], axis=0)
            else:
                # Initial sampling
                print 'Initial sampling ...'
                self.is_exploiting = False
                self.center = self.init_point.flatten()
        
        # Choose when to exploit versus explore
        if np.all(mean[prob<=t]<0) or np.all(mean[prob<=t]>0):
            print 'Exploration stage ...'
            if self.is_exploiting:
                self.is_exploiting = False
                dist_l = np.linalg.norm(self.D-self.center, axis=1)
                self.last = self.D[np.argmax(dist_l)]
                                   
            if 1-nu < 0 or (mu+self.margin)/(1-nu)**.5 > t: # k_m <= 1, delta_explore has a feasible solution
                beta = (2*math.log((mu**2+t**2*nu)/
                                   (t*(mu**2+(self.eta**2-1)*self.margin**2*nu)**.5-self.margin*mu)))**.5
                self.delta_explore = ls*beta
                candidates = gen_candidates(self.pool_size, self.last, self.delta_explore)
                mean, var = self.model.predict_mean_var(candidates)
                prob = (np.abs(mean)+self.margin)/np.sqrt(var)
            else:
                self.delta_explore = ls
                # In the case where delta_explore does not have a feasible solution, 
                # we need to manually set its value such that the query strategy has 
                # feasible solutions in the local pool
                # (i.e., there are candidates with p_epsilon > tau)
                while True:
                    candidates = gen_candidates(self.pool_size, self.last, self.delta_explore)
                    mean, var = self.model.predict_mean_var(candidates)
                    prob = (np.abs(mean)+self.margin)/np.sqrt(var)
                    
                    # Check if there are feasible solutions
                    if sum(prob<=t) > 0:
                        break
                    else:
                        self.delta_explore *= 1.5
                        
            print 'delta_explore =', self.delta_explore
            
        else:
            print 'Exploitation stage ...'
            self.is_exploiting = True
            print 'delta_exploit =', self.delta_exploit

        ind = prob <= t # constraint
        
        # If no feasible solution, look for feasible solutions around the farthest labeled point instead
        if sum(ind) == 0:
            print 'Exploration stage ...'
            self.is_exploiting = False
            dist_l = np.linalg.norm(self.D-self.center, axis=1)
            self.last = self.D[np.argmax(dist_l)]
            candidates = gen_candidates(self.pool_size, self.last, self.delta_explore)
            
            mean, var = self.model.predict_mean_var(candidates)
            prob = (np.abs(mean)+self.margin)/np.sqrt(var)
            ind = prob <= t # constraint
        
        obj1 = var
        obj2 = np.linalg.norm(candidates-self.center, axis=1)
        obj = obj1**self.is_exploiting * obj2**(1-self.is_exploiting)
        score = obj[ind]
        print np.min(score)
        
        new_query = candidates[ind][np.argmin(score)].reshape((1,-1))
        self.D = np.append(self.D, new_query, axis=0)
        self.last = new_query
        self.candidates = candidates

        return new_query
        