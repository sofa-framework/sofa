import Sofa

# Python version of the "oneParticleSample" in cpp located in applications/tutorials/oneParticle
def oneParticleSample(node):
 node.name='oneParticleSample'
 node.gravity=[0.0, -9.81, 0.0]
 solver = node.createObject('EulerSolver', name='solver')
 solver.printLog = 'false'
 node.addObject(solver);
 particule_node = node.createChild('particle_node')
 particle = particule_node.createObject('MechanicalObject',name='particle')
 particle.resize(1)
 mass = particule_node.createObject('UniformMass',name='mass')
 mass.mass=1.0
 return 0

