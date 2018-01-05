import math as M
import numpy as np
import matplotlib.pyplot as plt
from CircAvg2D import get_weighted_angle, get_angle, get_angle_between, vec_eeq, eeq,\
    subd_4PT_one_step, subd_LR_one_step

#------------------------------------------------------------------------------
def line_intersect(p0, p1, m0=None, m1=None, q0=None, q1=None):
    ''' intersect 2 lines given 2 points and 
        (either associated slopes or one extra point)
        Inputs:
        p0 - first point of first line [x,y]
        p1 - fist point of second line [x,y]
        m0 - slope of first line
        m1 - slope of second line
        q0 - second point of first line [x,y]
        q1 - second point of second line [x,y]
    '''
    if m0 is  None:
        if q0 is None:
            raise ValueError('either m0 or q0 is needed')
        dy = q0[1] - p0[1]
        dx = q0[0] - p0[0]
        lhs0 = [-dy, dx]
        rhs0 = p0[1] * dx - dy * p0[0]
    else:
        lhs0 = [-m0, 1]
        rhs0 = p0[1] - m0 * p0[0]

    if m1 is  None:
        if q1 is None:
            raise ValueError('either m1 or q1 is needed')
        dy = q1[1] - p1[1]
        dx = q1[0] - p1[0]
        lhs1 = [-dy, dx]
        rhs1 = p1[1] * dx - dy * p1[0]
    else:
        lhs1 = [-m1, 1]
        rhs1 = p1[1] - m1 * p1[0]

    a = np.array([lhs0, 
                  lhs1])

    b = np.array([rhs0, 
                  rhs1])
    try:
        px = np.linalg.solve(a, b)
    except:
        px = np.array([np.nan, np.nan])

    return px

#------------------------------------------------------------------------------
def create_circle(p0, p1, n0):
    m0 = n0[1]/n0[0] if not eeq(n0[0], 0) else None
    if eeq(p1[0], p0[0]):
        m1 = 0.0
    elif eeq(p1[1], p0[1]):
        m1 = None
    else:
        m1 = (p1[1] - p0[1]) / (p1[0] - p0[0])
        m1 = -1.0 / m1

    if m0 == m1:
        cntr = None
        radius = None
    else:  
        q0 = p0 + n0 if m0 == None else None
        mid_pt = (p0 + p1)/2.0
        q1 = mid_pt + np.array([0.,1.]) if m1 == None else None
        cntr = line_intersect(p0, mid_pt, m0 = m0, m1 = m1, q0 = q0, q1 = q1)
        radius = np.linalg.norm(p0 - cntr)
    return cntr, radius

#------------------------------------------------------------------------------
def create_circle_v2(p0, p1, n0):
    p1p0_vec = p1-p0
    p1p0_len = np.linalg.norm(p1p0_vec)
    p1p0_vec /= p1p0_len
    alpha = get_angle_between(n0, p1p0_vec)
    alpha = M.pi - alpha
    radius = (p1p0_len/2.)/M.cos(alpha)
    cntr = p0 - n0*radius
    return cntr, radius

#------------------------------------------------------------------------------
def create_circle_v3(p, n, q):
    natural_norm = p-q
    radius = np.linalg.norm(natural_norm)
    if eeq(radius,0.):
        center = p
        radius = 0.
    else:
        center = p + natural_norm
        natural_norm /= radius
        center = q if vec_eeq(natural_norm,n) else center
    return center, radius 

#------------------------------------------------------------------------------
def find_circle_center(p0, p1, radius, old_center):
    if eeq(radius,0.) or radius < 0.0:
        return old_center
    mid_pt = (p0 + p1)/2.0
    mid_perp = np.array([-(p0 - p1)[1], (p0 - p1)[0]])
    segm_length = np.linalg.norm(p0-p1)
    if eeq(segm_length,0.):
        return p0
    mid_perp /= np.linalg.norm(mid_perp)
    sqr_length_to_center = radius ** 2 - (segm_length/2.) ** 2
    if sqr_length_to_center < 0. or eeq(sqr_length_to_center, 0.0):
        print 'Negative value'
    length_to_center = sqr_length_to_center ** 0.5
    cntr1 = mid_pt + mid_perp * length_to_center
    cntr2 = mid_pt - mid_perp * length_to_center
    test1a = np.linalg.norm(p0 - cntr1)
    test1b = np.linalg.norm(p1 - cntr1)
    test2a = np.linalg.norm(p0 - cntr2)
    test2b = np.linalg.norm(p1 - cntr2)
    d1 = np.linalg.norm(old_center - cntr1)
    d2 = np.linalg.norm(old_center - cntr2)
    return (cntr1 if d1 < d2 else cntr2)

#-----------------------------------------------------------------------------
def alt_average_export( t1, t2, b_open, p1, p2, n1, n2 ):
    p, n = alt_average(t1, p1,p2,n1,n2)
    return p,n, p, 0,0,0
#------------------------------------------------------------------------------
def alt_average(t, p0, p1, n0, n1):
    c0,r0 = create_circle(p0, p1, n0)
    c1,r1 = create_circle(p1, p0, n1)
    
    #new_cntr = c0*(1. - t) + c1*t    
    #-- v1
    # new_radius = r0*(1. - t) + r1*t
    #-- v2
    #d0 = np.linalg.norm(p0 - new_cntr)
    #d1 = np.linalg.norm(p1 - new_cntr)
    #new_radius = d0*(1. - t) + d1*t
    #-- v3
    res_radius = r0*(1. - t) + r1*t
    
    #draw_debug = True
    draw_debug = False

    res_cntr0 = find_circle_center(p0, p1, res_radius, c0)
    if res_cntr0[0] is None or res_cntr0[0] == np.nan:
        draw_debug = True
        print 'Num problem'
    vec0_p0 = p0 - res_cntr0
    if eeq(np.linalg.norm(vec0_p0), 0.):
        draw_debug = True
        print 'Num problem'
    vec0_p0 /= np.linalg.norm(vec0_p0)
    vec0_p1 = p1 - res_cntr0
    vec0_p1 /= np.linalg.norm(vec0_p1)
    res_ang0 = get_weighted_angle(1. - t, t, vec0_p0, vec0_p1)
    res_cntr1 = find_circle_center(p0, p1, res_radius, c1)
    vec1_p0 = p0 - res_cntr1
    vec1_p0 /= np.linalg.norm(vec1_p0)
    vec1_p1 = p1 - res_cntr1
    vec1_p1 /= np.linalg.norm(vec1_p1)
    res_ang1 = get_weighted_angle(1. - t, t, vec1_p0, vec1_p1)
    
    interp_ang = get_weighted_angle(1. - t, t, n0, n1)
    if abs(res_ang0 - interp_ang) < abs(res_ang1 - interp_ang):
       res_ang = res_ang0 
       res_cntr = res_cntr0
    else:
       res_ang = res_ang1
       res_cntr = res_cntr1
    res_norm = np.array([M.cos(res_ang), M.sin(res_ang)])
    res_pt = res_cntr + res_norm * res_radius
    res_ang = interp_ang
    res_norm = np.array([M.cos(res_ang), M.sin(res_ang)])


    if draw_debug:
        ar0 = plt.Arrow( p0[0], p0[1], n0[0], n0[1], width=0.03, fc='b', ec='none' )
        ar1 = plt.Arrow( p1[0], p1[1], n1[0], n1[1], width=0.03, fc='g', ec='none' )
        pt0 = plt.Circle( (p0[0], p0[1]), radius=0.02, fc='b', ec='none')
        pt1 = plt.Circle( (p1[0], p1[1]), radius=0.02, fc='g', ec='none')
        cr0 = plt.Circle( (c0[0], c0[1]), radius=r0, fc='none', ec='b')
        cr1 = plt.Circle( (c1[0], c1[1]), radius=r1, fc='none', ec='g')
        crr = plt.Circle( (res_cntr[0], res_cntr[1]), radius=res_radius, fc='none', ec='r')
        arr = plt.Arrow( res_pt[0], res_pt[1], res_norm[0], res_norm[1], width=0.03, fc='r', ec='none' )
        ptr = plt.Circle( (res_pt[0], res_pt[1]), radius=0.02, fc='r', ec='none')
        plt.gca().add_patch(ar0)
        plt.gca().add_patch(ar1)
        plt.gca().add_patch(pt0)
        plt.gca().add_patch(pt1)
        plt.gca().add_patch(cr0)
        plt.gca().add_patch(cr1)
        plt.gca().add_patch(crr)
        plt.gca().add_patch(arr)
        plt.gca().add_patch(ptr)
        plt.axis('scaled')
        plt.show()

    return res_pt, res_norm

#-----------------------------------------------------------------------------
def alt_average_v2(t, p0, p1, n0, n1):

    np0, np1, nn0, nn1, gamma, ofx, ofy = put_to_zero(p0, p1, n0, n1)
    
    c0,r0 = create_circle_v2(np0, np1, nn0)
    c1,r1 = create_circle_v2(np1, np0, nn1)
    
    res_radius = r0*(1. - t) + r1*t
    res_cntr = find_circle_center(np0, np1, res_radius, c0)
    vec_p0 = np0 - res_cntr
    vec_p0 /= np.linalg.norm(vec_p0)
    vec_p1 = np1 - res_cntr
    vec_p1 /= np.linalg.norm(vec_p1)
    res_ang2 = get_weighted_angle(1. - t, t, vec_p0, vec_p1)
    res_ang = get_weighted_angle(1. - t, t, nn0, nn1)
    theta = get_angle_between( nn0, nn1 )
    res_ang1 = theta*(1.-t)
    res_norm = np.array([M.cos(res_ang), M.sin(res_ang)])
    res_pt = res_cntr + res_norm * res_radius

    #draw_debug = True
    draw_debug = False

    if draw_debug:
        ar0 = plt.Arrow( np0[0], np0[1], nn0[0], nn0[1], width=0.03, fc='b', ec='none' )
        ar1 = plt.Arrow( np1[0], np1[1], nn1[0], nn1[1], width=0.03, fc='g', ec='none' )
        pt0 = plt.Circle( (np0[0], np0[1]), radius=0.02, fc='b', ec='none')
        pt1 = plt.Circle( (np1[0], np1[1]), radius=0.02, fc='g', ec='none')
        cr0 = plt.Circle( (c0[0], c0[1]), radius=r0, fc='none', ec='b')
        cr1 = plt.Circle( (c1[0], c1[1]), radius=r1, fc='none', ec='g')
        crr = plt.Circle( (res_cntr[0], res_cntr[1]), radius=res_radius, fc='none', ec='r')
        arr = plt.Arrow( res_pt[0], res_pt[1], res_norm[0], res_norm[1], width=0.03, fc='r', ec='none' )
        ptr = plt.Circle( (res_pt[0], res_pt[1]), radius=0.02, fc='r', ec='none')
        plt.gca().add_patch(ar0)
        plt.gca().add_patch(ar1)
        plt.gca().add_patch(pt0)
        plt.gca().add_patch(pt1)
        plt.gca().add_patch(cr0)
        plt.gca().add_patch(cr1)
        plt.gca().add_patch(crr)
        plt.gca().add_patch(arr)
        plt.gca().add_patch(ptr)
        plt.axis('scaled')
        plt.show()

    res_pt, res_norm = put_back(res_pt, res_norm, gamma, ofx, ofy )
    p0, n0 = put_back(p0, n0, gamma, ofx, ofy )
    p1, n1 = put_back(p1, n1, gamma, ofx, ofy )

    return res_pt, res_norm

#-----------------------------------------------------------------------------
def put_to_zero(p0, p1, n0, n1):
    offset_x = p0[0]
    offset_y = p0[1]
    gamma = p1-p0
    p1p0_len = np.linalg.norm(gamma)
    gamma /= p1p0_len

    cw_rot_mtrx = np.array([[ gamma[0], gamma[1]],
                            [-gamma[1], gamma[0]]])
    nn0 = cw_rot_mtrx.dot(n0)
    nn1 = cw_rot_mtrx.dot(n1)
    np0 = np.array([0., 0.])
    np1 = np.array([p1p0_len, 0.])
    return np0, np1, nn0, nn1, gamma, offset_x, offset_y

#-----------------------------------------------------------------------------
def put_back(npt, nnorm, gamma, offset_x, offset_y):
    cw_rot_mtrx = np.array([[ gamma[0], -gamma[1]],
                            [ gamma[1],  gamma[0]]])
    norm = cw_rot_mtrx.dot(nnorm)
    pt = cw_rot_mtrx.dot(npt) + np.array([offset_x, offset_y])
    return pt, norm

#-----------------------------------------------------------------------------
def sliding_circ_average_export( t1, t2, b_open, p1, p2, n1, n2 ):
    p, n = sliding_circ_average(t1, p1,p2,n1,n2)
    return p,n, p, 0,0,0

#-----------------------------------------------------------------------------
def sliding_circ_average(t, p0, p1, n0, n1):
    m0 = n0[1]/n0[0] if not eeq(n0[0], 0) else None
    m1 = n1[1]/n1[0] if not eeq(n1[0], 0) else None
    if m0 == m1:
        res_ang = get_weighted_angle(1. - t, t, n0, n1)
        res_norm = np.array([M.cos(res_ang), M.sin(res_ang)])
        res_pt = p0*(1. - t) + p1*t
    else:  
        q0 = p0 + n0 if m0 == None else None
        q1 = p1 + n1 if m1 == None else None
        q = line_intersect(p0, p1, m0 = m0, m1 = m1, q0 = q0, q1 = q1)
        c0, r0 = create_circle_v3(p0, n0, q)
        c1, r1 = create_circle_v3(p1, n1, q)
        res_radius = r0*(1. - t) + r1*t
        res_cntr = c0*(1. - t) + c1*t
        res_ang = get_weighted_angle(1. - t, t, n0, n1)
        res_norm = np.array([M.cos(res_ang), M.sin(res_ang)])
        res_pt = res_cntr + res_norm * res_radius
    
    #draw_debug = True
    draw_debug = False

    if draw_debug:
        ar0 = plt.Arrow( p0[0], p0[1], n0[0], n0[1], width=0.03, fc='b', ec='none' )
        ar1 = plt.Arrow( p1[0], p1[1], n1[0], n1[1], width=0.03, fc='g', ec='none' )
        pt0 = plt.Circle( (p0[0], p0[1]), radius=0.02, fc='b', ec='none')
        pt1 = plt.Circle( (p1[0], p1[1]), radius=0.02, fc='g', ec='none')
        cr0 = plt.Circle( (c0[0], c0[1]), radius=r0, fc='none', ec='b')
        cr1 = plt.Circle( (c1[0], c1[1]), radius=r1, fc='none', ec='g')
        crr = plt.Circle( (res_cntr[0], res_cntr[1]), radius=res_radius, fc='none', ec='r')
        arr = plt.Arrow( res_pt[0], res_pt[1], res_norm[0], res_norm[1], width=0.03, fc='r', ec='none' )
        ptr = plt.Circle( (res_pt[0], res_pt[1]), radius=0.02, fc='r', ec='none')
        plt.gca().add_patch(ar0)
        plt.gca().add_patch(ar1)
        plt.gca().add_patch(pt0)
        plt.gca().add_patch(pt1)
        plt.gca().add_patch(cr0)
        plt.gca().add_patch(cr1)
        plt.gca().add_patch(crr)
        plt.gca().add_patch(arr)
        plt.gca().add_patch(ptr)
        plt.axis('scaled')
        plt.show()

    return res_pt, res_norm
#-----------------------------------------------------------------------------
def plot_pts_and_norms(pts, nrm, b_open, draw_norms, clr, 
                       linestyle='', linewidth=1.0, cnvs = plt ):
    n = len(pts)
    nn = n-1 if b_open else n
    for i in range(nn):
        curr_pt = pts[i]
        next_pt = pts[(i+1)%n]
        if linestyle.startswith('da'):
            cnvs.plot([curr_pt[0], next_pt[0]], 
                        [curr_pt[1], next_pt[1]], 
                        color=clr, linestyle=linestyle,
                        linewidth=linewidth, dashes=(1,15))
        else:
            cnvs.plot([curr_pt[0], next_pt[0]], 
                        [curr_pt[1], next_pt[1]], 
                        color=clr, linestyle=linestyle, 
                        linewidth=linewidth)

        if draw_norms:
            curr_norm = nrm[i]
            #colr = clr if i != 2 else 'r'
            colr = clr
            #curr_norm /= 2.0 #if i == 2 else 1.0
            gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                               curr_norm[0], curr_norm[1], 
                               width=0.05, fc=colr, ec='none')
            cnvs.gca().add_patch(gnr)
    if draw_norms and b_open:
        curr_norm = nrm[-1]
        curr_pt = pts[-1]
        gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                           curr_norm[0], curr_norm[1], 
                           width=linewidth, fc=clr, ec='none')
        cnvs.gca().add_patch(gnr)

#-----------------------------------------------------------------------------
def get_bisector(prev_pt, curr_pt, next_pt):
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt
    v1 = np.array([v1[1],-v1[0]])
    v2 = np.array([v2[1],-v2[0]])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    bisec = get_weighted_angle(0.5, 0.5, v1, v2)
    bisec = np.array([M.sin(bisec), -M.cos(bisec)])
    return bisec
   
#-----------------------------------------------------------------------------
def get_limit_norms(pts):
    limit_norms = []
    n_of_pts = len(pts)
    for i in range(n_of_pts):
        curr_pt = pts[i]
        prev_pt = pts[(i-1+n_of_pts)%n_of_pts]       
        next_pt = pts[(i+1)%n_of_pts]
        curr_norm = get_bisector(prev_pt,curr_pt,next_pt)
        limit_norms.append(curr_norm)
    return limit_norms

#-----------------------------------------------------------------------------
def subd_test():
    pts = []
    norms = []
    a_radius = 2
    b_radius = 1
    n_steps =  4
    b_open = False
    one_step = 2. * M.pi / n_steps
    for i in range(n_steps):
        t = one_step*i + M.pi/4.
        curr_pt = np.array([a_radius * M.cos(t), b_radius * M.sin(t)])
        curr_norm = np.array([b_radius * M.cos(t), a_radius * M.sin(t)])
        curr_norm /= np.linalg.norm(curr_norm)
        pts.append(curr_pt)
        norms.append(curr_norm)

    n_steps =  100
    one_step = 2. * M.pi / n_steps
    orig_pts = []
    for i in range(n_steps):
        t = one_step*i
        curr_pt = np.array([a_radius * M.cos(t), b_radius * M.sin(t)])
        orig_pts.append(curr_pt)
    
    n_of_iterations = 5
    average_fn = alt_average_export
    #average_fn = alt_average_v2
    #average_fn = sliding_circ_average_export

    slid_mlr_pts = pts[:]
    slid_mlr_norms = norms[:]
    slid_m4p_pts = pts[:]
    slid_m4p_norms = norms[:]
    for i in range(n_of_iterations):
        slid_mlr_pts, slid_mlr_norms = subd_LR_one_step(slid_mlr_pts, 
                                                        slid_mlr_norms, 
                                                        b_open, 
                                                        average_fn, 
                                                        1)
        #slid_m4p_pts, slid_m4p_norms = subd_4PT_one_step(slid_m4p_pts, 
        #                                                slid_m4p_norms, 
        #                                                b_open, 
        #                                                average_fn)
 
    limit_mlr_norms = get_limit_norms(slid_mlr_pts)
        
    plt.axis('equal')
    plot_pts_and_norms(orig_pts, norms, b_open, False, clr='k', 
                       linewidth=0.3, linestyle='solid')    
    plot_pts_and_norms(pts, norms, b_open, False, clr='k', 
                       linewidth=0.3, linestyle='solid')    
    plot_pts_and_norms(slid_mlr_pts, slid_mlr_norms, b_open, True, clr='b', 
                       linewidth=0.3, linestyle='solid')    
    plot_pts_and_norms(slid_mlr_pts, limit_mlr_norms, b_open, True, clr='g', 
                       linewidth=0.3, linestyle='solid')    
    #plot_pts_and_norms(slid_m4p_pts, slid_m4p_norms, b_open, True, clr='r', 
    #                   linewidth=1.0, linestyle='solid')    
    plt.show()

#-----------------------------------------------------------------------------
def stand_alone_pair_test():
    p0 = np.array([0.,0.])
    p1 = np.array([1.,0.])
    n0 = np.array([-(2.**0.5)/2.,-(2.**0.5)/2.])
    n1 = np.array([(3.**0.5)/2., 0.5])

    average_fn = alt_average
    #average_fn = alt_average_v2
    #average_fn = sliding_circ_average

    #pt_1, norm_1 = average_fn(1.0, p0, p1, n0, n1)
    #pt05, norm05 = average_fn(0.5, p0, p1, n0, n1)
    #pt025, norm025 = average_fn(0.25, p0, p1, n0, n1)
 
    #pt05_05, norm05_05 = average_fn(0.5, p0, pt05, n0, norm05)

    pts = []
    norms = []
    n_steps = 30
    one_step = 1. / n_steps
    for i in range(n_steps+1):
        t = one_step*i
        curr_angle = 2. * M.pi * t
        n1 = np.array([M.cos(curr_angle), M.sin(curr_angle)])
        curr_pt, curr_norm = average_fn(0.5, p0, p1, n0, n1)
        #curr_pt, curr_norm = average_fn(t, p0, p1, n0, n1)
        pts.append(curr_pt)
        norms.append(curr_norm)

    ar0 = plt.Arrow( p0[0], p0[1], n0[0], n0[1], width=0.03, fc='b', ec='none' )
    ar1 = plt.Arrow( p1[0], p1[1], n1[0], n1[1], width=0.03, fc='g', ec='none' )
    pt0 = plt.Circle( (p0[0], p0[1]), radius=0.02, fc='b', ec='none')
    pt1 = plt.Circle( (p1[0], p1[1]), radius=0.02, fc='g', ec='none')
    #cr0 = plt.Circle( (c0[0], c0[1]), radius=r0, fc='none', ec='b')
    #cr1 = plt.Circle( (c1[0], c1[1]), radius=r1, fc='none', ec='g')

    plt.gca().add_patch(ar0)
    plt.gca().add_patch(ar1)
    plt.gca().add_patch(pt0)
    plt.gca().add_patch(pt1)
    #plt.gca().add_patch(cr0)
    #plt.gca().add_patch(cr1)
    for i in range(len(pts)):
        curr_pt = pts[i]
        curr_norm = norms[i]
        #gr_pt = plt.plot([curr_pt[0],next_pt[0]], [curr_pt[1], next_pt[1]], 'r' )
        gr_nr = plt.Arrow(curr_pt[0], curr_pt[1], curr_norm[0], curr_norm[1], 
                  width=0.03, fc='r', ec='none' )
        #plt.gca().add_patch(gr_pt)
        plt.gca().add_patch(gr_nr)
    
    plt.axis('scaled')
    plt.show()

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    subd_test()
    #stand_alone_pair_test()
#============================= END OF FILE ====================================
