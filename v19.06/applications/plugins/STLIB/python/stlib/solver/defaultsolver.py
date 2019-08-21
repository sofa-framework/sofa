# -*- coding: utf-8 -*-

def DefaultSolver(node, iterative=True):
    '''
    Adds EulerImplicit, CGLinearSolver

    Components added:
        EulerImplicit
        CGLinearSolver
    '''
    node.createObject('EulerImplicit', name='TimeIntegrationSchema')
    if iterative:
        return node.createObject('CGLinearSolver', name='LinearSolver')

    return node.createObject('SparseLDLSolver', name='LinearSolver')

### This function is just an example on how to use the DefaultHeader function. 
def createScene(rootNode):
	DefaultSolver(rootNode) 	
    
