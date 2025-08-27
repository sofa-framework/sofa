import drjit
from drjit.auto import Float, Array3f, UInt

drjit.set_log_level(drjit.LogLevel.Debug)

radius = Float(0.5)
print(radius) # [1, 2, 3, 4]

center = Array3f([0,0,0])
print(center)

positions = Array3f([1,2,3,4],0,0)
print(positions)

print("Version 1", drjit.sqrt((positions - center)**2 ) - radius )  
print("Version 2", drjit.sqrt((positions - center)**2 ) - radius )  

drjit.whos()