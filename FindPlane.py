import numpy as np
import scipy.optimize
import functools

#-----------------------------------------------------------------------------
def plane(n, points, params):
    a = params[0]
    b = params[1]
    c = params[2]
    p0 = points[0]
    p1 = points[1]
    p = np.array([p1[0] - p0[0], 
                  p1[1] - p0[1],
                  p1[2] - p0[2]])
    n_cross_p = np.cross(n,p)
    candidate_norm = np.array([a,b,c])
    penalty = candidate_norm.dot(n_cross_p)
    return penalty

#-----------------------------------------------------------------------------
def error(params, points, normals):
    result = 0
    for n in normals:
        norm_diff = plane(n, points, params)
        result += norm_diff**2
    return result
#-----------------------------------------------------------------------------
def get_best_plane(p0, p1, n0, n1):
    points = [(p0[0],p0[1],p0[2]),
              (p1[0],p1[1],p1[2])]
    normals = [(n0[0],n0[1],n0[2]),
               (n1[0],n1[1],n1[2])]
    fun = functools.partial(error, points=points, normals=normals)
    params0 = [n0[0],n0[1],n0[2]]
    res = scipy.optimize.minimize(fun, params0)
    a = res.x[0]
    b = res.x[1]
    c = res.x[2]
    return np.array([a,b,c])
#========================== END OF FILE ======================================
