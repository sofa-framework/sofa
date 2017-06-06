
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GLU import *

from contextlib import contextmanager
import math

# a few helpers for debugging
@contextmanager
def debug(**kwargs):
    '''setup gl for drawing debug stuff: no lightingm, fat lines/points, etc'''

    line_width = kwargs.get('line_width', 5)
    point_size = kwargs.get('point_size', 8)    
    lighting = kwargs.get('lighting', False)
    
    glEnable(GL_COLOR_MATERIAL)

    if not lighting: glDisable(GL_LIGHTING)

    # TODO pushattrib
    glLineWidth(line_width)
    glPointSize(point_size)
    
    try:
        yield
    finally:
        # TODO popattrib
        glLineWidth(1.0)
        glPointSize(1.0)

        if not lighting: glEnable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)


@contextmanager        
def frame(rigid):
    '''setup a gl frame from a Compliant.types.Rigid3'''
    glPushMatrix()
    glTranslate(*rigid.center)
    axis, angle = rigid.orient.axis_angle()

    if angle:
        glRotate(angle / math.pi * 180, *axis)

    try:
        yield
    finally:
        glPopMatrix()
        

