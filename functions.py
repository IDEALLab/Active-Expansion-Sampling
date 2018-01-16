"""
Test functions for classification (feasible domain identification)

Author(s): Wei Chen (wchen459@umd.edu)
"""


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def one_circle(D, p=0, s=0):
    
    landmark = np.array([[2, 0]])
    threshold = 1.5
    L = (np.min(pairwise_distances(D, landmark), axis=1) < threshold)
    L = L*2-1
    L *= (-1)**np.random.binomial(1, p, D.shape[0]) # add noise
    return L
    
def two_circles(D, p=0, s=0):
    
    landmark = np.array([[0, 0], 
                         [3, 0]])
    threshold = 1.
    L = (np.min(pairwise_distances(D, landmark), axis=1) < threshold)
    L = L*2-1
    L *= (-1)**np.random.binomial(1, p, D.shape[0]) # add noise
    return L
    
def branin(D, p=0, s=0, sigma=1.0):
    
    x1 = D[:,0]
    x2 = D[:,1]
    g = (x2-5.1*x1**2/4/np.pi**2+5/np.pi*x1-6)**2 + 10*(1-1/8/np.pi)*np.cos(x1) + 10 # branin
    g += s * np.random.normal(0, sigma**2, D.shape[0]) # add Gaussian noise
    y1 = g <= 8
    y2 = np.logical_and(x1>-9, x1<14)
    y3 = np.logical_and(x2>-7, x2<17)
    y = np.logical_and(np.logical_and(y1,y2), y3)
    y = y*2-1
    y *= (-1)**np.random.binomial(1, p, D.shape[0]) # add Bernoulli noise
    return y

def hosaki(D, p=0, s=0, sigma=1.0):
    
    x1 = D[:,0]
    x2 = D[:,1]
    g = (1.-8.*x1+7.*x1**2.-7./3*x1**3+x1**4/4.)*x2**2*np.exp(-x2)
    g += s * np.random.normal(0, sigma**2, D.shape[0]) # add Gaussian noise
    y = np.logical_and(g<=-1., x2>0)*2-1
    y *= (-1)**np.random.binomial(1, p, D.shape[0]) # add Bernoulli noise
    return y
    
def beam(D, p=0, s=0, sigma=1.0):
    
    b = D[:,0]
    h = D[:,1]
    It = (b**3*h+b*h**3)/12
    Iz = b**3*h/12
    Iy = b*h**3/12
    y0 = b*h <= 0.0025 # cross sectional area
    y1 = 5*.5**3/(3*216620*Iy) <= 5 # maximum tip deflection
    y2 = 6*5*.5/(b*h**2) <= 240000 # bending stress
    y3 = 3*5/(2*b*h) <= 120000 # shear stress
    y4 = h/b <= 10 # aspect ratio
    y5 = b/h <= 10
    y6 = 4/.5**2*(86.65*It*216.62*Iz/(1-.27**2))**.5 >= 2*5/1e6 # failue force of buckling
    y7 = b > 0
    y8 = h > 0
    y = np.logical_and.reduce((y0,y1,y2,y3,y4,y5,y6,y7,y8))
    y = y*2-1
    return y
    
def two_spheres(D, p=0):
    
    landmark = np.zeros((2, D.shape[1]))
    landmark[1,0] += 3.0
    threshold = 1.
    L = np.min(pairwise_distances(D, landmark), axis=1) < threshold
    L = L*2-1
    L *= (-1)**np.random.binomial(1, p, D.shape[0]) # add noise
    return L