import math
import sys

import Sofa

from SofaPython import Quaternion
import Compliant.StructuralAPI



def setShowDofs(dofs, scale=0.5, mode=0):
    dofs.showObject = True
    dofs.showObjectScale = scale
    dofs.drawMode = mode



def createScene(rootNode):

    rootNode.gravity="0 0 0"

    rootNode.createObject('RequiredPlugin', pluginName="Compliant")

    rootNode.createObject('CompliantAttachButtonSetting' )
    rootNode.createObject('CompliantImplicitSolver', name='odesolver', stabilization="true")

    rootNode.createObject('LDLTSolver', name='numsolver')
    rootNode.createObject('LDLTResponse', name='response')


    # no necessary for mechanical but only for visual debugging
    targetPos="1 3 1  0.4619397662556433 -0.19134171618254492 0.4619397662556433 0.7325378163287419"
    target = rootNode.createObject("MechanicalObject",template="Rigid3d",name="target_(red)", position=targetPos)
    setShowDofs(target,.6,2)



    objectFull = Compliant.StructuralAPI.RigidBody(rootNode, "objectFullToTarget_(green)")
    objectFull.setManually(offset=[0,2,0,0,0,0,1])
    setShowDofs(objectFull.dofs,.5,1)
    constraintNode = objectFull.node.createChild("fullconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromTargetMapping",targets=targetPos)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")


    objectRotation = Compliant.StructuralAPI.RigidBody(rootNode, "objectRotationToTarget_(blue)")
    objectRotation.setManually(offset=[0,2,0,0,0,0,1])
    setShowDofs(objectRotation.dofs,.5,3)
    constraintNode = objectRotation.node.createChild("rotationconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromTargetMapping",targets=targetPos,translation=False)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")


    objectTranslation = Compliant.StructuralAPI.RigidBody(rootNode, "objectTranslationToTarget_(purple)")
    objectTranslation.setManually(offset=[0,2,0,0,0,0,1])
    setShowDofs(objectTranslation.dofs,.5,5)
    constraintNode = objectTranslation.node.createChild("translationconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromTargetMapping",targets=targetPos,rotation=False)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")



    ##########################


    objectFull = Compliant.StructuralAPI.RigidBody(rootNode, "objectFullToWorldFrame_(green)")
    objectFull.setManually(offset=[-1,-1,0,0.4619397662556433, -0.19134171618254492, 0.4619397662556433, 0.7325378163287419])
    setShowDofs(objectFull.dofs,.5,1)
    constraintNode = objectFull.node.createChild("fullconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromWorldFrameMapping",targets=targetPos)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")


    objectRotation = Compliant.StructuralAPI.RigidBody(rootNode, "objectRotationToWorldFrame_(blue)")
    objectRotation.setManually(offset=[-1,-1,0,0.4619397662556433, -0.19134171618254492, 0.4619397662556433, 0.7325378163287419])
    setShowDofs(objectRotation.dofs,.5,3)
    constraintNode = objectRotation.node.createChild("rotationconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromWorldFrameMapping",targets=targetPos,translation=False)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")


    objectTranslation = Compliant.StructuralAPI.RigidBody(rootNode, "objectTranslationToWorldFrame_(purple)")
    objectTranslation.setManually(offset=[-1,-1,0,0.4619397662556433, -0.19134171618254492, 0.4619397662556433, 0.7325378163287419])
    setShowDofs(objectTranslation.dofs,.5,5)
    constraintNode = objectTranslation.node.createChild("translationconstraint")
    constraintNode.createObject("MechanicalObject",template="Vec6d")
    constraintNode.createObject("RigidJointFromWorldFrameMapping",targets=targetPos,rotation=False)
    constraintNode.createObject("UniformCompliance",compliance="1e-10")