import sofa.core
import sofa.simulation

rootNode = sofa.simulation.Simulation.GetRoot()
mo = rootNode.createObject("MechanicalObject")
print "dof:", mo.getPathName()
print mo.findData("position")
