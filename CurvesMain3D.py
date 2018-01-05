import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math as M
from CircAvg3D import *

#-----------------------------------------------------------------------------
def curves_main():
    n_of_iterations = 5
    #n_of_pts = 10
    #orig_pts, orig_nrms, b_open = create_helix_polyline(n_of_pts, -M.pi, M.pi)
    orig_pts, orig_nrms, b_open = create_input_on_a_polygon5()

    mlr_pts, mlr_nrms = orig_pts[:], orig_nrms[:]
    m4pt_pts, m4pt_nrms = orig_pts[:], orig_nrms[:]
    l4pt_pts, l4pt_nrms = orig_pts[:], orig_nrms[:]
    for _ in range(n_of_iterations):
        mlr_pts, mlr_nrms = subd_LR_one_step(mlr_pts, mlr_nrms, 
                                             b_open, circle_avg_3D)
        m4pt_pts, m4pt_nrms = subd_4PT_one_step(m4pt_pts, m4pt_nrms, 
                                                b_open, circle_avg_3D)
        l4pt_pts, l4pt_nrms = subd_4PT_one_step(l4pt_pts, l4pt_nrms, 
                                                b_open, linear_avg)

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=0, elev=90)
    plot_pts_and_norms(orig_pts, orig_nrms, b_open, True, label = 'Input',
                       clr='k', linewidth=0.4, linestyle='solid', cnvs=ax)
    plot_pts_and_norms(mlr_pts, mlr_nrms, b_open, False, label = 'MLR Result',
                       clr='b', linewidth=1.0, linestyle='solid',cnvs=ax)
    plot_pts_and_norms(m4pt_pts, m4pt_nrms, b_open, False, label = 'M4Pt Result',
                       clr='r', linewidth=1.0, linestyle='solid',cnvs=ax)
    #plot_pts_and_norms(l4pt_pts, l4pt_nrms, b_open, False, label = 'L4Pt Result',
    #                   clr='g', linewidth=0.6, linestyle='solid',cnvs=ax)
    ax.legend()
    plt.show()

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    curves_main()
#============================= END OF FILE ===================================