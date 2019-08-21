# -*- coding: utf-8 -*-
"""
Scene debuging facilities.

"""
import Sofa
import SofaPython
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from splib.numerics import *

SofaPython.__SofaPythonEnvironment_modulesExcludedFromReload.append("OpenGL.GL")
SofaPython.__SofaPythonEnvironment_modulesExcludedFromReload.append("OpenGL.GLU")
SofaPython.__SofaPythonEnvironment_modulesExcludedFromReload.append("OpenGL.GLUT")

debugManager=None
def DebugManager(parentNode):
    global debugManager
    ImmediateRenderer(parentNode)
    return parentNode

currentImmediateRenderer = None
def drawText(text, x, y):
    global currentImmediateRenderer
    if currentImmediateRenderer == None:
        return
    currentImmediateRenderer.addText(text,x,y)

def drawLine(p0,p1):
    global currentImmediateRenderer
    if currentImmediateRenderer == None:
        return
    currentImmediateRenderer.addEdge(p0, p1)

def worldToScreenPoint(p):
    return gluProject(p[0],p[1],p[2], currentImmediateRenderer.mvm,
                      currentImmediateRenderer.pm, currentImmediateRenderer.viewport)


class BluePrint(Sofa.PythonScriptController):
        def __init__(self, node):
            self.name = "BluePrintController"
            self.rules = []
            self.circles = []

        def addRule(self, origin=[0.0,0.0,0.0], direction=[1.0,0.0,0.0], spacing=1.0, length=10, text="cm"):
            self.rules.append([origin,direction,spacing,length,text])

        def addCircle(self, origin, radius):
            self.circles((origin,radius))

        def drawCircle(self, o, r):
            for i in 10:
                currentImmediateRenderer.addEdge( vvadd( o, vsmul(d, i)),
                                                  vvadd( o, vsmul(d, i+1)))


        def drawRule(self, o,d,s,l,t):
            global currentImmediateRenderer
            """Emits the drawing code needed to visualize the rule"""
            step = l / s
            for i in range(0, int(step)):
                currentImmediateRenderer.addEdge( vvadd( o, vsmul(d, i)),
                                                  vvadd( o, vsmul(d, i+1)))
                currentImmediateRenderer.addPoint( vvadd( o, vsmul(d, i)) )

        def onBeginAnimationStep(self, s):
            for rule in self.rules:
                self.drawRule(rule[0], rule[1], rule[2], rule[3], rule[4])


        def onIdle(self):
            for rule in self.rules:
                self.drawRule(rule[0], rule[1], rule[2], rule[3], rule[4])



class ImmediateRenderer(Sofa.PythonScriptController):
    def __init__(self, rootNode):
        global currentImmediateRenderer
        self.name = "DebugManager"
        self.edges = []
        self.edges2D = []
        self.textes = []
        self.points = []
        currentImmediateRenderer = self
        glutInit()
        self.mvm = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.pm = glGetDoublev(GL_PROJECTION_MATRIX)
        self.viewport = glGetInteger( GL_VIEWPORT )


    def addText(self, text, x, y):
        self.textes.append([text,int(x),int(y)])

    def addEdge(self, p0, p1):
        self.edges.append([p0,p1])

    def addPoint(self, p0):
        self.points.append(p0)

    def addEdge2D(self, p0, p1):
        self.edges2D.append([p0,p1])


    def addRenderable(self, r):
        self.renderable.append(r)

    def drawAll2D(self):
        viewport = glGetInteger( GL_VIEWPORT );

        glDepthMask(GL_FALSE)

        glPushAttrib( GL_LIGHTING_BIT )
        glPushAttrib( GL_ENABLE_BIT )
        glEnable( GL_COLOR_MATERIAL )
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, viewport[2], 0, viewport[3] )

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        self.drawAllText()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glPopAttrib()
        glPopAttrib()

        glDepthMask(GL_TRUE)

    def drawAllText(self):
        for text in self.textes:
            glRasterPos2i( text[1], text[2] )
            glColor(1.0,0.0,0.0)
            for c in text[0]:
                glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(c))

    def onBeginAnimationStep(self, dt):
        self.textes = []
        self.edges = []
        self.points = []

    def onIdle(self):
        self.textes = []
        self.edges = []
        self.points = []

    def draw(self):
        self.mvm = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.pm = glGetDoublev(GL_PROJECTION_MATRIX)
        self.viewport = glGetInteger( GL_VIEWPORT )

        self.drawAll2D()

        glDisable(GL_LIGHTING)
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glColor3f(0.8,0.7,1.0)
        for e in self.edges:
            glVertex3dv(e[0])
            glVertex3dv(e[1])
        glEnd()

        glPointSize(5.0)
        glColor3d(1.0,1.0,1.0)
        glBegin(GL_POINTS)
        for p in self.points:
            glVertex3dv(p)

        glEnd()

class TracerLog(object):
    def __init__(self, filename):
        self.outfile = open(filename, "wt")

    def writeln(self, s):
        self.outfile.write(s+"\n")

    def close(self):
        self.outfile.close()


def kwargs2str(kwargs):
    s=""
    for k in kwargs:
        s+=", "+k+"="+repr(kwargs[k])
    return s

class Tracer(object):

    def __init__(self, node, backlog, depth, context):
        self.node = node
        self.backlog = backlog
        self.depth = depth
        self.context = context

    def createObject(self, type, **kwargs):
        self.backlog.writeln(self.depth+self.node.name+".createObject('"+type+"' "+kwargs2str(kwargs)+")")
        n = self.node.createObject(type, **kwargs)
        return n

    def createChild(self, name, **kwargs):
        self.backlog.writeln("")
        self.backlog.writeln(self.depth+"#========================= "+name+" ====================== ")
        self.backlog.writeln(self.depth+name+" = "+self.node.name+".createChild('"+name+"' "+kwargs2str(kwargs)+")")
        n = Tracer(self.node.createChild(name, **kwargs), self.backlog, self.depth, name)
        return n

    def getObject(self, name):
        n = self.node.getObject(name)
        return n

    def addObject(self, tgt):
        self.backlog.writeln(self.depth+self.node.name+".addObject('"+tgt.name+"')")
        if isinstance(tgt, type(Tracer)):
            return self.node.addObject(tgt.node)
        return self.node.addObject(tgt)

    def __getattr__(self, value):
        return self.node.__getattribute__(value)
