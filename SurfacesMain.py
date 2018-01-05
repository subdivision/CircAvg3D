import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from DCELCore import DCtrlMesh
from CircAvg3D import circle_avg_3D
from CSubd2D import linear_avg
#from BSplineAvg import bspline_average_export_3D

INP_PATH_PREFIX = 'C:/TAU/InputMeshes/'
RES_PATH_PREFIX = 'C:/TAU/DebugMeshes/'

#-----------------------------------------------------------------------------
def plot_results(orig_ctrl_mesh, circ_avg_ctrl_mesh, lin_ctrl_mesh):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d', aspect='equal')
    ax.view_init(azim=0, elev=0)

    if orig_ctrl_mesh:
        orig_ctrl_mesh.plot(ax, True, 'k')
    if circ_avg_ctrl_mesh:
        circ_avg_ctrl_mesh.plot(ax, False, 'r')
    if lin_ctrl_mesh:
        lin_ctrl_mesh.plot(ax, False, 'b')

    plt.show()

#-----------------------------------------------------------------------------
def avg_fn_to_str(avg_fn):
    return '_circ_' if avg_fn == circle_avg_3D else '_lin_'

#-----------------------------------------------------------------------------
def create_tower_mesh(id, avg_func):
    file_prefix = 'tower_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_quad_cube(10.)
    orig_ctrl_mesh.extrude_face(9, 20.)
    orig_ctrl_mesh.extrude_face(30, 20.)
    orig_ctrl_mesh.extrude_face(34, 20.)
    orig_ctrl_mesh.set_naive_normals()
    #orig_ctrl_mesh.print_ctrl_mesh()
    return orig_ctrl_mesh, file_prefix

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d', aspect='equal')
    ax.view_init(azim=30, elev=25)
    orig_ctrl_mesh.plot(ax, True, 'k')
    plt.show()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_cube4_mesh(id, avg_func):
    file_prefix = 'cube_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_quad_cube(10.)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_cube3_mesh(id, avg_func):
    file_prefix = 'cube_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_triang_cube(10.)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_torus4_mesh(id, avg_func, n_of_verts_in_init_torus):
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(False, n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_torus3_mesh(id, avg_func, n_of_verts_in_init_torus):
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(True, n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tri_mesh_stl_file(id, avg_func, file_name):
    file_prefix = file_name + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    inp_file = INP_PATH_PREFIX + file_name
    orig_ctrl_mesh.init_as_triang_mesh_stl_file(inp_file)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def srf_main():
    n_of_iterations = 3
    init_vrts_torus = 6
    #plot_debug = True
    plot_debug = False

    #ref_method, ref_name = DCtrlMesh.refine_as_catmull_clark, 'cc_'
    #ref_method, ref_name = DCtrlMesh.refine_as_kob4pt, 'kob4pt_'
    #ref_method, ref_name = DCtrlMesh.refine_as_butterfly, 'butterfly_'
    ref_method, ref_name = DCtrlMesh.refine_as_loop, 'loop_'

    res_file_suffix = ref_name + str(n_of_iterations) + 'iters.off'

    #circ_avg_ctrl_mesh, circ_res_name = create_tower_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_cube4_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_torus4_mesh(2, circle_avg_3D, init_vrts_torus)
    #circ_avg_ctrl_mesh, circ_res_name = create_torus3_mesh(2, circle_avg_3D, init_vrts_torus)
    #circ_avg_ctrl_mesh, circ_res_name = create_cube3_mesh(2, circle_avg_3D)
    circ_avg_ctrl_mesh, circ_res_name = create_tri_mesh_stl_file(2, circle_avg_3D, 'fox.stl')
 
    #lin_ctrl_mesh, lin_res_name = create_tower_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_cube4_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_torus4_mesh(3, linear_avg, init_vrts_torus)
    #lin_ctrl_mesh, lin_res_name = create_torus3_mesh(3, linear_avg, init_vrts_torus)
    #lin_ctrl_mesh, lin_res_name = create_cube3_mesh(3, linear_avg)
    lin_ctrl_mesh, lin_res_name = create_tri_mesh_stl_file(3, linear_avg, 'fox.stl')

    circ_res_name += res_file_suffix
    lin_res_name += res_file_suffix

    for i in range(n_of_iterations):
        circ_avg_ctrl_mesh = ref_method(circ_avg_ctrl_mesh)
        lin_ctrl_mesh = ref_method(lin_ctrl_mesh)
       
    circ_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + circ_res_name)
    lin_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + lin_res_name)
    if plot_debug:
        plot_results(lin_ctrl_mesh2, None, lin_ctrl_mesh)
    
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    srf_main()
#============================= END OF FILE ===================================