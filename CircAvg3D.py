import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math as M
from CSubd2D import *

#-----------------------------------------------------------------------------
def create_helix_polyline(mesh_size, low_bnd, upr_bnd ):
    step = (upr_bnd - low_bnd) / (mesh_size - 1);
    pnts = []
    norms = []
    b_open = True
    for i in range(mesh_size):
        t = low_bnd + i * step;
        x = M.cos( t )
        y = M.sin( t )
        z = t
        pnts.append( np.array([x,y,z]))
        dx = -1.0 * M.sin(t)
        dy =  M.cos(t)
        dz =  1
        first_der = np.array([dx, dy, dz])
        ddx = -1.0 * M.cos(t)
        ddy = -1.0 * M.sin(t)
        ddz = 0
        second_der = np.array([ddx, ddy, ddz])
        bi_norm = np.cross(first_der, second_der)
        res_norm = np.cross(bi_norm, first_der)
        res_norm /= np.linalg.norm(res_norm)
        norms.append(-res_norm)

    return pnts, norms, b_open

#-----------------------------------------------------------------------------
def create_input_on_a_polygon5():
    b_open = False
    c30 = (3.0**0.5)/2.0
    pts = [np.array([-2., -2.,  -1.]),
           np.array([ 4., -3.,  1.]),
           np.array([ 2.,  0.,  0.]),
           np.array([ 5.,  3.5, -1.]),
           np.array([-1.,  3.,  1.])]

    nrm = [np.array([-1., -1., 0.]),
           np.array([ 1., -1., 0.1]),
           #np.array([  c30, -0.5, 0.5]), #rd
           #np.array([ c30, 0.5]), #ru
           np.array([ -c30, 0.5, 0.5]), #lu
           #np.array([ -c30, -0.5]), #ld
           np.array([ 1.,  1., -0.1]),
           np.array([-1.,  1., 0])]
    nnrm = []
    for n in nrm:
        nnrm.append( n/np.linalg.norm(n))
    return pts, nnrm, b_open 

#-----------------------------------------------------------------------------
def create_input_on_a_polygon5_in_plane():
    b_open = False
    c30 = (3.0**0.5)/2.0
    pts = [np.array([-2., -2. ]),
           np.array([ 4., -3. ]),
           np.array([ 2.,  0. ]),
           np.array([ 5.,  3.5]),
           np.array([-1.,  3. ])]

    nrm = [np.array([-1., -1.]),
           np.array([ 1., -1.]),
           #np.array([  c30, -0.5, 0.5]), #rd
           #np.array([ c30, 0.5]), #ru
           np.array([ -c30, 0.5]), #lu
           #np.array([ -c30, -0.5]), #ld
           np.array([ 1.,  1.]),
           np.array([-1.,  1.])]
    nnrm = []
    for n in nrm:
        nnrm.append( n/np.linalg.norm(n))
    return pts, nnrm, b_open 

#-----------------------------------------------------------------------------
def project_vector_to_plane(anchor_pt, plane_norm, vctr):
    target_pt = anchor_pt + vctr
    length = np.dot(plane_norm, vctr)
    delta_vec = length * plane_norm
    result_pt = target_pt - delta_vec
    result_vec = result_pt - anchor_pt
    result_vec /= np.linalg.norm(result_vec)
    return result_vec

#-----------------------------------------------------------------------------
def project_point_to_plane(anchor_pt, plane_norm, query_pt):
    connecting_vec = query_pt - anchor_pt
    length = np.dot(plane_norm, connecting_vec)
    delta_vec = length * plane_norm
    result_pt = query_pt - delta_vec
    
    conn_vec_tag = result_pt - anchor_pt
    test_dot = np.dot(plane_norm, conn_vec_tag)
    
    #pln_d0 = -anchor_pt.dot(plane_norm)
    #pln_xx, pln_yy = np.meshgrid(range(-10,10), range(-10,10))
    #pln_zz0 = (-plane_norm[0] * pln_xx - plane_norm[1] * pln_yy - pln_d0) * 1.\
    #          / plane_norm[2]
    #plt3d = plt.figure().gca(projection='3d')
    #plt3d.view_init(azim=0, elev=90)
    #plt3d.plot_surface(pln_xx, pln_yy, pln_zz0, color='b', alpha=0.3)
    #plt3d.quiver([anchor_pt[0]], [anchor_pt[1]], [anchor_pt[2]],
    #             [plane_norm[0]], [plane_norm[1]], [plane_norm[2]], 
    #             color='b', pivot='tail')
    #plt3d.plot([query_pt[0]], [query_pt[1]], [query_pt[2]], 'go')
    #plt3d.plot([result_pt[0]], [result_pt[1]], [result_pt[2]], 'ro')
    #plt3d.quiver([result_pt[0]], [result_pt[1]], [result_pt[2]],
    #             [plane_norm[0]], [plane_norm[1]], [plane_norm[2]], length=length, 
    #             color='r', pivot='tail')
    #plt.show()    
    
    return result_pt

#-----------------------------------------------------------------------------
def get_vect_in_coord_sys(v, transform_matrix):
    res = transform_matrix.dot(v)
    return res

#-----------------------------------------------------------------------------
def circle_avg_3D(t0, t1, b_open, p0, p1, n0, n1):
    x_axis = np.array([1.,0.,0.])
    y_axis = np.array([0.,1.,0.])
    new_z = np.cross(n0, n1)
    if vec_eeq(new_z, np.array([0.,0.,0.])):# and not vec_eeq(n0, n1):
        #print 'Num inst.'
        return linear_avg(t0, t1, b_open, p0, p1, n0, n1)
    new_z /= np.linalg.norm(new_z)
    new_y = np.cross(new_z, x_axis)
    new_y /= np.linalg.norm(new_y)
    new_x = np.cross(new_y, new_z)
    new_x /= np.linalg.norm(new_x)

    direct_transf = np.array([[new_x[0], new_x[1], new_x[2]],
                              [new_y[0], new_y[1], new_y[2]],
                              [new_z[0], new_z[1], new_z[2]]])
    p1_moved = p1 - p0
    orig_seg_length = np.linalg.norm(p1_moved)
    p0_tag = np.array([.0,.0,.0])
    p1_tag = project_point_to_plane(p0_tag, new_z, p1_moved)
    new_p1 = get_vect_in_coord_sys(p1_tag, direct_transf)
    #new_p1 = get_vect_in_coord_sys(p1_moved, direct_transf)
    new_n0 = get_vect_in_coord_sys(n0, direct_transf)
    new_n0 /= np.linalg.norm(new_n0)
    new_n1 = get_vect_in_coord_sys(n1, direct_transf)
    new_n1 /= np.linalg.norm(new_n1)

    p0_2D = np.array([.0,.0])
    p1_2D = np.array([new_p1[0],new_p1[1]])
    n0_2D = np.array([new_n0[0],new_n0[1]])
    n1_2D = np.array([new_n1[0],new_n1[1]])

    res_pt_2D, res_norm_2D, \
    cntr_2D, radius_2D, beta1, beta2 = circle_avg(t0, t1, 
                                              b_open, p0_2D, p1_2D, 
                                              n0_2D, n1_2D)
    z_plane_offset = p1_moved.dot(new_z)
    z_plane_offset *= t1
    res_pt_pln = np.array([res_pt_2D[0], res_pt_2D[1], z_plane_offset ]) 
    res_norm_pln = np.array([res_norm_2D[0], res_norm_2D[1], 0. ]) 
    cntr_pt_pln = np.array([cntr_2D[0], cntr_2D[1], 0. ]) 
    back_transf = np.linalg.inv(direct_transf)
    res_pt = get_vect_in_coord_sys(res_pt_pln, back_transf)
    res_pt += p0 
    cntr_pt = get_vect_in_coord_sys(cntr_pt_pln, back_transf)
    cntr_pt += p0 
    res_norm = get_vect_in_coord_sys(res_norm_pln, back_transf)

    #pln_d0 = -p0.dot(new_z)
    #pln_d1 = -p1.dot(new_z)
    #pln_xx, pln_yy = np.meshgrid(range(-15,15), range(-15,15))
    #pln_zz0 = (-new_z[0] * pln_xx - new_z[1] * pln_yy - pln_d0) * 1. /new_z[2]
    #pln_zz1 = (-new_z[0] * pln_xx - new_z[1] * pln_yy - pln_d1) * 1. /new_z[2]
    #plt3d = plt.figure().gca(projection='3d')
    #plt3d.view_init(azim=0, elev=90)
    #plt3d.plot_surface(pln_xx, pln_yy, pln_zz0, color='b', alpha=0.3)
    #plt3d.plot_surface(pln_xx, pln_yy, pln_zz1, color='g', alpha=0.3)
    #plt3d.quiver([p0[0]], [p0[1]], [p0[2]],
    #             [n0[0]], [n0[1]], [n0[2]], 
    #             color='b', pivot='tail')
    #plt3d.quiver([p0[0]], [p0[1]], [p0[2]],
    #             [n1[0]], [n1[1]], [n1[2]], 
    #             color='g', pivot='tail')
    #plt3d.quiver([p1[0]], [p1[1]], [p1[2]],
    #             [n1[0]], [n1[1]], [n1[2]], 
    #             color='g', pivot='tail')
    #plt3d.quiver([p1[0]], [p1[1]], [p1[2]],
    #             [n0[0]], [n0[1]], [n0[2]], 
    #             color='b', pivot='tail')
    #plt3d.quiver([res_pt[0]], [res_pt[1]], [res_pt[2]],
    #             [res_norm[0]], [res_norm[1]], [res_norm[2]], 
    #             color='r', pivot='tail')
    #plt3d.quiver([p0[0]], [p0[1]], [p0[2]],
    #             [new_z[0]], [new_z[1]], [new_z[2]], 
    #             color='m', pivot='tail')
    #plt3d.quiver([p1[0]], [p1[1]], [p1[2]],
    #             [new_z[0]], [new_z[1]], [new_z[2]], 
    #             color='m', pivot='tail')
    #plt.show()   



    return res_pt, res_norm, cntr_pt,  radius_2D, beta1, beta2 


#-----------------------------------------------------------------------------
def plot_pts_and_norms(pts, nrm, b_open, draw_norms, clr, label='a curve',
                       linestyle='', linewidth=1.0, cnvs = plt ):
    n = len(pts)
    xs = [pts[i][0] for i in range(n) ]
    ys = [pts[i][1] for i in range(n) ]
    zs = [pts[i][2] for i in range(n) ]
    if not b_open:
        xs.append(pts[0][0])
        ys.append(pts[0][1])
        zs.append(pts[0][2])
    cnvs.plot(xs, ys, zs, label=label, 
            color=clr, linestyle=linestyle, 
            linewidth=linewidth)
    if draw_norms:
        us = [nrm[i][0] for i in range(n) ]
        vs = [nrm[i][1] for i in range(n) ]
        ws = [nrm[i][2] for i in range(n) ]
        if not b_open:
            xs = xs[:-1]
            ys = ys[:-1]
            zs = zs[:-1]
        cnvs.quiver(xs,ys,zs,us,vs,ws, color=clr, pivot='tail')

#============================= END OF FILE ===================================