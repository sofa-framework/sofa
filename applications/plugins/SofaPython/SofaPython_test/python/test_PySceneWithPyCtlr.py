# -*- coding: utf-8 -*-

import Sofa

import os

currentdir = os.path.dirname(__file__)


def createScene(rootNode):

    # This create a PythonScriptController that permits to programatically implement new behavior 
    # or interactions using the Python programming langage. The controller is referring to a 
    # file named "controller.py". 
    rootNode.createObject('PythonScriptController', filename=os.path.join(currentdir,"testController.py"), classname="controller")
