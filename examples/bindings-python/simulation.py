import sofa.core
import sofa.simulation

rootNode = sofa.simulation.Simulation.GetRoot()
mo = rootNode.createObject("MechanicalObject", position="1 2 3")
print "dof:", mo.getPathName()
print mo.findData("position")
