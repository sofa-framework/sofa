# -*- coding: utf-8 -*-
from splib.objectmodel import SofaPrefab

@SofaPrefab
class Interaction(object):
    """
    Store a list of mechanical object to interact with

    Args:

        targets ([objects])     the object to interact with and that don't have a solver

    """

    def __init__(self, parent, targets):
        self.node = parent.createChild("Interaction")
        self.node.createObject("EulerImplicitSolver")
        self.node.createObject("CGLinearSolver")
        for target in targets:
            self.node.addChild(target)
