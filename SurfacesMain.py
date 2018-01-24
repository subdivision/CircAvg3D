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
def create_tetrahedron3_mesh(id, avg_func):
    file_prefix = 'tetrahedron_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_tetrahedron()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tetrahedron4_mesh(id, avg_func):
    file_prefix = 'tetrahedron_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_tetrahedron()
    orig_ctrl_mesh = orig_ctrl_mesh.refine_as_catmull_clark(\
        get_edge_vertex_func = DCtrlMesh.get_edge_vertex_as_mid,
        get_vrtx_vertex_func = DCtrlMesh.get_vrtx_vertex_as_copy)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tower4_mesh(id, avg_func):
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
def create_tower3_mesh(id, avg_func):
    file_prefix = 'tower_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh, _ = create_tower4_mesh(id - 1, avg_func)
    orig_ctrl_mesh = orig_ctrl_mesh.triangulize_quad_mesh()
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
def create_torus4_mesh(id, avg_func):
    global n_of_verts_in_init_torus
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(False, n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_torus3_mesh(id, avg_func):
    global n_of_verts_in_init_torus
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(True, n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_mesh3_stl_file(id, avg_func):
    global stl_file_name
    file_prefix = file_name + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    inp_file = INP_PATH_PREFIX + file_name
    orig_ctrl_mesh.init_as_triang_mesh_stl_file(inp_file)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_mesh4_stl_file(id, avg_func):
    global stl_file_name
    file_prefix = file_name + '_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    inp_file = INP_PATH_PREFIX + file_name
    orig_ctrl_mesh.init_as_triang_mesh_stl_file(inp_file)
    orig_ctrl_mesh = orig_ctrl_mesh.refine_as_catmull_clark(\
        get_edge_vertex_func = DCtrlMesh.get_edge_vertex_as_mid,
        get_vrtx_vertex_func = DCtrlMesh.get_vrtx_vertex_as_copy)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def get_initial_mesh():
    n_of_verts_in_init_torus = 6
    stl_file_name = 'fox.stl'
    #stl_file_name = 'bunny.stl'
    #stl_file_name = 'cube.stl'

    #--- Circle Average Mesh ------------------------------
    circ_avg_ctrl_mesh, circ_res_name = create_tower4_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_cube4_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_torus4_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_mesh4_stl_file(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_tetrahedron4_mesh(2, circle_avg_3D)

    #circ_avg_ctrl_mesh, circ_res_name = create_tower3_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_cube3_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_torus3_mesh(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_mesh3_stl_file(2, circle_avg_3D)
    #circ_avg_ctrl_mesh, circ_res_name = create_tetrahedron3_mesh(2, circle_avg_3D)
 
    #--- Linear Average Mesh ------------------------------
    lin_ctrl_mesh, lin_res_name = create_tower4_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_cube4_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_torus4_mesh(3, linear_avg, init_vrts_torus)
    #lin_ctrl_mesh, lin_res_name = create_mesh4_stl_file(3, linear_avg, stl_file_name)
    #lin_ctrl_mesh, lin_res_name = create_tetrahedron4_mesh(3, linear_avg)

    #lin_ctrl_mesh, lin_res_name = create_tower3_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_torus3_mesh(3, linear_avg, init_vrts_torus)
    #lin_ctrl_mesh, lin_res_name = create_cube3_mesh(3, linear_avg)
    #lin_ctrl_mesh, lin_res_name = create_mesh3_stl_file(3, linear_avg, stl_file_name)
    #lin_ctrl_mesh, lin_res_name = create_tetrahedron3_mesh(3, linear_avg)

    return circ_avg_ctrl_mesh, circ_res_name, lin_ctrl_mesh, lin_res_name 
#-----------------------------------------------------------------------------
def srf_main():
    n_of_iterations = 3

    #ref_method, ref_name = DCtrlMesh.refine_as_catmull_clark, 'cc_'
    #ref_method, ref_name = DCtrlMesh.refine_as_kob4pt, 'kob4pt_'
    ref_method, ref_name = DCtrlMesh.refine_as_butterfly, 'butterfly_'
    #ref_method, ref_name = DCtrlMesh.refine_as_loop, 'loop_'

    res_file_suffix = ref_name + str(n_of_iterations) + 'iters.off'
    circ_avg_ctrl_mesh, circ_res_name, \
        lin_ctrl_mesh, lin_res_name = get_initial_mesh()

    circ_res_name += res_file_suffix
    lin_res_name += res_file_suffix

    for i in range(n_of_iterations):
        circ_avg_ctrl_mesh = ref_method(circ_avg_ctrl_mesh)
        lin_ctrl_mesh = ref_method(lin_ctrl_mesh)
       
    circ_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + circ_res_name)
    lin_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + lin_res_name)

    
#-----------------------------------------------------------------------------
def rotate_normals():
    n_of_iterations = 3
    #ref_method, ref_name = DCtrlMesh.refine_as_catmull_clark, 'cc_'
    ref_method, ref_name = DCtrlMesh.refine_as_kob4pt, 'kob4pt_'
    #ref_method, ref_name = DCtrlMesh.refine_as_butterfly, 'butterfly_'
    #ref_method, ref_name = DCtrlMesh.refine_as_loop, 'loop_'

    for w in np.linspace(0, 1, 6):
        circ_avg_ctrl_mesh, circ_res_name, _, _ = get_initial_mesh()
    
        circ_avg_ctrl_mesh.init_normals(np.array([1., 1., 1.]))
        circ_avg_ctrl_mesh.set_naive_normals(w)

        res_file_suffix = ref_name + 'wei_' + str(w) + '_'\
                          + str(n_of_iterations)\
                          + 'iters.off'
        circ_curr_res_name = circ_res_name + res_file_suffix

        for i in range(n_of_iterations):
            circ_avg_ctrl_mesh = ref_method(circ_avg_ctrl_mesh)
       
        circ_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + circ_curr_res_name)
    
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    #srf_main()
    rotate_normals()
#============================= END OF FILE ===================================