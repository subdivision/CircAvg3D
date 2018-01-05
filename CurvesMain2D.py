import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math as M
from CircAvg3D import *

#-----------------------------------------------------------------------------
def create_input_on_a_rectang_2D():
    b_open = False
    c45 = (2.**0.5)/2.0
    pts = [np.array([-1.,  3. ]),
           np.array([-1., -3. ]),
           np.array([ 1., -3. ]),
           np.array([ 1.,  3. ])]

    nrm = [np.array([ -c45,  c45]),
           np.array([ -c45, -c45]),
           np.array([  c45, -c45]),
           np.array([  c45,  c45])]
    return pts, nrm, b_open 

#-----------------------------------------------------------------------------
def plot_pts_and_norms_2D(pts, nrm, b_open, draw_norms, clr, 
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
            #wdth = 0.15 if i != 2 else 0.3
            wdth = 0.3
            #colr = clr if i != 2 else 'r'
            colr = clr
            #curr_norm /= 2.0 #if i == 2 else 1.0
            gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                               curr_norm[0], curr_norm[1], 
                               width=0.2, fc=colr, ec='none' )
            cnvs.gca().add_patch(gnr)
            if clr != 'k':
                colr = 'b' if clr == 'r' else 'y'
                prev_pt = pts[(i-1+n)%n]
                pseudo_n1 = next_pt - curr_pt
                pseudo_n1 /= np.linalg.norm(pseudo_n1)
                pseudo_n2 = prev_pt - curr_pt
                pseudo_n2 /= np.linalg.norm(pseudo_n2)
                bisec_t = get_weighted_angle(0.5, 0.5, pseudo_n1, pseudo_n2)
                bisec_vec = -np.array([M.cos(bisec_t), M.sin(bisec_t)])
                gnr2 = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                                   bisec_vec[0], bisec_vec[1], 
                                   width=0.2, fc=colr, ec='none' )
                cnvs.gca().add_patch(gnr2)


    if draw_norms and b_open:
        curr_norm = nrm[-1]
        curr_pt = pts[-1]
        gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                           curr_norm[0], curr_norm[1], 
                           width=0.03, fc=clr, ec='none' )
        cnvs.gca().add_patch(gnr)
        

#-----------------------------------------------------------------------------
def curves_main():
    n_of_iterations = 5
    orig_pts, orig_nrms, b_open = create_input_on_a_rectang_2D()

    mlr_pts, mlr_nrms = orig_pts[:], orig_nrms[:]
    mlr2_pts, mlr2_nrms = orig_pts[:], orig_nrms[:]
    m4pt_pts, m4pt_nrms = orig_pts[:], orig_nrms[:]
    #l4pt_pts, l4pt_nrms = orig_pts[:], orig_nrms[:]
    for _ in range(n_of_iterations):
        #mlr_pts, mlr_nrms = subd_LR_one_step(mlr_pts, mlr_nrms, 
        #                                     b_open, circle_avg_v2, n_deg=1)
        #mlr2_pts, mlr2_nrms = subd_LR_one_step(mlr2_pts, mlr2_nrms, 
        #                                     b_open, circle_avg, n_deg=1)
        m4pt_pts, m4pt_nrms = subd_4PT_one_step(m4pt_pts, m4pt_nrms, 
                                                b_open, circle_avg)
        #l4pt_pts, l4pt_nrms = subd_4PT_one_step(l4pt_pts, l4pt_nrms, 
        #                                        b_open, linear_avg)

    #frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)
    plot_pts_and_norms_2D(orig_pts, orig_nrms, b_open, True, 
                       clr='k', linewidth=0.4, linestyle='dotted')
    #plot_pts_and_norms_2D(mlr_pts, mlr_nrms, b_open, True, 
    #                   clr='r', linewidth=0.6, linestyle='solid')
    #plot_pts_and_norms_2D(mlr2_pts, mlr2_nrms, b_open, True, 
    #                   clr='m', linewidth=0.6, linestyle='solid')
    plot_pts_and_norms_2D(m4pt_pts, m4pt_nrms, b_open, True,
                       clr='r', linewidth=0.6, linestyle='solid')
    #plot_pts_and_norms_2D(l4pt_pts, l4pt_nrms, b_open, False, label = 'L4Pt Result',
    #                   clr='g', linewidth=0.6, linestyle='solid',cnvs=ax)
    plt.axis('scaled')
    plt.show()

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    curves_main()
#============================= END OF FILE ===================================