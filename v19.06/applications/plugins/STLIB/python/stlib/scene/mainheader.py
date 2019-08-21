# -*- coding: utf-8 -*-
import Sofa
from splib.animation import AnimationManager

def MainHeader(node, gravity=[0.0, -9.8, 0.0], dt=0.01, plugins=[], repositoryPaths=[], doDebug=False):
    '''
    Args:
        gravity (vec3f): define the gravity vector.

        dt  (float): define the timestep.

        plugins (list str): list of plugins to load

        repositoryPaths (list str): list of path to the specific data repository

    Structure:
        .. sourcecode:: qml

            rootNode : {
                gravity : gravity,
                dt : dt,
                VisualStyle,
                RepositoryPath,
                RequiredPlugin,
                OglSceneFrame,
                FreeMotionAnimationLoop,
                GenericConstraintSolver,
                DiscreteIntersection
            }

    '''
    node.createObject('VisualStyle')
    node.findData('gravity').value=gravity;
    node.findData('dt').value=dt

    if not isinstance(plugins, list):
        Sofa.msg_error("MainHeader", "'plugins' expected to be a list, got "+str(type(plugins)))
        return node

    if "SofaMiscCollision" not in plugins:
        plugins.append("SofaMiscCollision")

    if "SofaPython" not in plugins:
        plugins.append("SofaPython")

    confignode = node.createChild("Config")
    for name in plugins:
        confignode.createObject('RequiredPlugin', name=name, printLog=False)

    i=0
    for repository in repositoryPaths:
        confignode.createObject('AddResourceRepository', name="AddResourceRepository"+str(i), path=repository)
        i+=1

    confignode.createObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    if doDebug:
        from splib.debug import DebugManager
        DebugManager(node)

    AnimationManager(node)
    return node


### This function is just an example on how to use the DefaultHeader function. 
def createScene(rootNode):
    import os
    MainHeader(rootNode, plugins=["SofaMiscCollision","SofaPython","SoftRobots"], repositoryPaths=[os.getcwd()])
