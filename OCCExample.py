"""
Simple example of bspline surface
"""

from OCC.gp import *
from OCC.Geom import *
from OCC.TColGeom import *
from OCC.TColgp import * 
from OCC.GeomConvert import *
from OCC.BRepBuilderAPI import *
from OCC.TopoDS import *
from OCC.STEPControl import *

from OCC.Display.SimpleGui import init_display
display, start_display, add_menu, add_function_to_menu = init_display()


def bezier_surfaces(event=None):
    """
    Generate and display bspline surface
    """
    display.EraseAll()
    array1 = TColgp_Array2OfPnt(1, 4, 1, 4)

    array1.SetValue(1, 1, gp_Pnt(1, 1, -1))
    array1.SetValue(1, 2, gp_Pnt(2, 1, 0))
    array1.SetValue(1, 3, gp_Pnt(3, 1, -1))
    array1.SetValue(1, 4, gp_Pnt(4, 1, 0))
    array1.SetValue(2, 1, gp_Pnt(1, 2, 3))
    array1.SetValue(2, 2, gp_Pnt(2, 2, 5))
    array1.SetValue(2, 3, gp_Pnt(3, 2, 2))
    array1.SetValue(2, 4, gp_Pnt(4, 2, 3))
    array1.SetValue(3, 1, gp_Pnt(1, 3, 2))
    array1.SetValue(3, 2, gp_Pnt(2, 3, 1))
    array1.SetValue(3, 3, gp_Pnt(3, 3, 0))
    array1.SetValue(3, 4, gp_Pnt(4, 3, 1))
    array1.SetValue(4, 1, gp_Pnt(1, 4, 0))
    array1.SetValue(4, 2, gp_Pnt(2, 4, -1))
    array1.SetValue(4, 3, gp_Pnt(3, 4, 0))
    array1.SetValue(4, 4, gp_Pnt(4, 4, -1))

    BZ1 = Geom_BezierSurface(array1)

    bezierarray = TColGeom_Array2OfBezierSurface(1, 1, 1, 1)
    bezierarray.SetValue(1, 1, BZ1.GetHandle())
    
    BB = GeomConvert_CompBezierSurfacesToBSplineSurface(bezierarray)
    if BB.IsDone():
        poles = BB.Poles().GetObject().Array2()
        uknots = BB.UKnots().GetObject().Array1()
        vknots = BB.VKnots().GetObject().Array1()
        umult = BB.UMultiplicities().GetObject().Array1()
        vmult = BB.VMultiplicities().GetObject().Array1()
        udeg = BB.UDegree()
        vdeg = BB.VDegree()

        BSPLSURF = Geom_BSplineSurface( poles, uknots, vknots, umult, vmult, udeg, vdeg, 0, 0 )
        BSPLSURF.Translate(gp_Vec(0,0,2))

    display.DisplayShape(BSPLSURF.GetHandle(), update=True)
    start_display()

if __name__ == '__main__':
    bezier_surfaces()
