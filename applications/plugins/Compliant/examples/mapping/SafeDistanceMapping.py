import math
import sys

import Sofa






def createScene(rootNode):

    rootNode.gravity="0 -10 0"
    rootNode.dt = 0.01

    rootNode.createObject('VisualStyle', displayFlags="showBehavior showMechanicalMappings showVisualModels" )


    rootNode.createObject('RequiredPlugin', pluginName="Compliant")

    rootNode.createObject('CompliantAttachButtonSetting',compliance=1e-6 )
    rootNode.createObject('CompliantImplicitSolver', stabilization=True, neglecting_compliance_forces_in_geometric_stiffness=False)
    rootNode.createObject('LDLTSolver', schur=False)




    rootNode.createObject("MechanicalObject", position="0 0 0  -2 3 4   1e-5 1e-7 -1e-6   -1e-5 -1e-7 1e-6    1 1 1", showObject=1, showObjectScale=.1, drawMode=1 )
    rootNode.createObject("UniformMass", mass="10000" )
    rootNode.createObject("UniformVelocityDampingForceField", dampingCoefficient="1000" )
    rootNode.createObject("FixedConstraint", indices="0" )






    distanceNode = rootNode.createChild("distance")
    distanceNode.createObject("MechanicalObject", template="Vec1d")
    distanceNode.createObject("SafeDistanceMapping", pairs="0 1  0 2  0 3", restLengths="0 1 0", showObjectScale=.01)
    distanceNode.createObject("UniformCompliance", resizable=True)




    distanceFromTargetNode = rootNode.createChild("distanceFromTarget")
    distanceFromTargetNode.createObject("MechanicalObject", template="Vec1d")
    distanceFromTargetNode.createObject("SafeDistanceFromTargetMapping", indices="4", restLengths="0", targets="2 2 2", showObjectScale=.01)
    distanceFromTargetNode.createObject("UniformCompliance", resizable=True)


