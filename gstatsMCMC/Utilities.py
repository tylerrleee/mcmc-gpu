import verde as vd
from scipy.spatial import KDTree
import numpy as np

def _interpolate(interp_method, fromx, fromy, data, tox, toy, k):
    # interpolate
    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        raise ValueError('the interp_method is not correctly defined, exit the function')
    
    interp.fit((fromx, fromy), data)
    result = interp.predict((tox, toy))
    
    return result

def min_dist_from_mask(xx, yy, mask):
    tree = KDTree(np.array([xx[mask], yy[mask]]).T)
    distance = tree.query(np.array([xx.ravel(), yy.ravel()]).T)[0].reshape(xx.shape)
    return distance