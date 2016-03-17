import sofa.core
import sofa.simulation

rootNode = sofa.simulation.Simulation.GetRoot()
mo = rootNode.createObject("MechanicalObject", position="1 2 3")
print "dof:", mo.getPathName()
position = mo.findData("position")
print position.getValueTypeString(), position
position.read("4 5 6")
print mo.findData("position")
