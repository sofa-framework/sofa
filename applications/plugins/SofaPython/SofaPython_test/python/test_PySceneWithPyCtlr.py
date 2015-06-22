# -*- coding: utf-8 -*-

import Sofa

import os

def createScene(rootNode):

    # This create a PythonScriptController that permits to programatically implement new behavior 
    # or interactions using the Python programming langage. The controller is referring to a 
    # file named "controller.py". 
    rootNode.createObject('PythonScriptController', filename="testController.py", classname="controller")
