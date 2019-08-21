# -*- coding: utf-8 -*-
"""
Templates for most of the common scene setups.

**Content:**

.. _sphinx_hyperlinks:

.. autosummary::

    Scene
    MainHeader
    ContactHeader
    Node
    Wrapper

|

stlib.scene.Scene
*****************

.. autoclass:: Scene
   :members:
   :undoc-members:

stlib.scene.Interaction
***********************

.. autoclass:: Interaction
   :members:
   :undoc-members:

stlib.scene.MainHeader
**********************

.. autofunction:: MainHeader

stlib.scene.ContactHeader
*************************

.. autofunction:: ContactHeader

stlib.scene.Node
****************

.. autofunction:: Node

stlib.scene.Wrapper
*******************

.. autoclass:: Wrapper
   :members:
   :undoc-members:
   :special-members: __getattr__

"""
from splib.objectmodel import SofaPrefab, SofaObject
from splib.scenegraph import get

from mainheader import MainHeader
from contactheader import ContactHeader
from stlib.solver import DefaultSolver
from interaction import Interaction

from wrapper import Wrapper

def Node(parentNode, name):
    """Create a new node in the graph and attach it to a parent node."""
    return parentNode.createChild(name)

@SofaPrefab
class Scene(SofaObject):
    """Scene(SofaObject)
    Create a scene with default properties.

       Arg:

        node (Sofa.Node)     the node where the scene will be attached

        gravity (vec3f)      the gravity of the scene

        dt (float)           the dt time

        plugins (list(str))  set of plugins that are used in this scene

        repositoryPath (list(str)) set of path where to read the data from

        doDebug (bool)       activate debugging facility (to print text)

       There is method to add default solver and default contact management
       on demand.
    """
    def __init__(self, node,  gravity=[0.0, -9.81, 0.0], dt=0.01, plugins=[], repositoryPaths=[], doDebug=False):
        self.node = node
        MainHeader(node, gravity=gravity, dt=dt, plugins=plugins, repositoryPaths=repositoryPaths, doDebug=doDebug)
        self.visualstyle = get(node, "VisualStyle")

    def addSolver(self):
        self.solver = DefaultSolver(self.node)

    def addContact(self,  alarmDistance, contactDistance, frictionCoef=0.0):
        ContactHeader(self.node,  alarmDistance, contactDistance, frictionCoef)
