Sofa Template Library
==========================

Utility functions and scene templates for the real-time simulation framework `Sofa <https://www.sofa-framework.org/>`_.
The different templates are organized in types and abstract the complexity of object
creation with `Sofa <https://www.sofa-framework.org/>`_.

The library is hosted on `github https://github.com/SofaDefrost/STLIB/` and it can be used with scenes
written in python and `PSL <https://github.com/sofa-framework/sofa/tree/master/applications/plugins/PSL>`_.


Example:


.. sourcecode:: python

    from stlib.scene import MainHeader
    from stlib.solver import DefaultSolver
    from stlib.physics.rigid import Cube, Sphere, Floor
    from stlib.physics.deformable import ElasticMaterialObject

    def createScene(rootNode):
        MainHeader(rootNode)
        DefaultSolver(rootNode)

        Sphere(rootNode, name="sphere", translation=[-5.0, 0.0, 0.0])
        Cube(rootNode, name="cube", translation=[5.0,0.0,0.0])

        ElasticMaterialObject(rootNode, name="dragon",
                              volumeMeshFileName="mesh/liver.msh",
                              surfaceMeshFileName="mesh/dragon.stl"
                              translation=[0.0,0.0,0.0])

        Floor(rootNode, name="plane", translation=[0.0, -1.0, 0.0])

**Content:**

.. toctree::
   :maxdepth: 2

.. automodule:: stlib


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`