# coding: utf8

import sys
import os
import numpy
import timeit
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

SofaRuntime.importPlugin("SofaAllCommonComponents")
                        
def oldSofa(obj):
        ol = obj.position.tolist()
        for i in range(len(ol)):
                ol[i] = [ol[i][0]+0.5, ol[i][1]+0.5,ol[i][2]+0.5]
        obj.position = ol


zeros = numpy.zeros((1000,3), dtype=numpy.float64)
ones = numpy.ones((1000,3), dtype=numpy.float64)
aList = zeros.tolist()

root = Sofa.Node("root")
obj = root.createObject("MechanicalObject", name="test", position=aList)
print("A counter: ", obj.position.getCounter())
obj.position += ones
print("B counter: ", obj.position.getCounter())
obj.position += ones
print("C counter: ", obj.position.getCounter())
d = obj.position
print("D counter: ", obj.position.getCounter())
d += ones
print("E counter: ", obj.position.getCounter())
d = obj.position + ones
print("F counter: ", obj.position.getCounter())

print(obj.position)
with obj.position.getWriteAccessor() as w:
        obj.position = w * zeros + ones
print("G counter: ", obj.position.getCounter())

print(obj.position)

def f(i, j, v):
        if i%3 == j:
                return 0
        return v

print("counter: ", obj.position.getCounter())
obj.position.apply(f)
print(obj.position)
print("counter: ", obj.position.getCounter())

### Ici 
obj.position[2:] = 5.0
print(obj.position)
print("counter: ", obj.position.getCounter())

obj.position[2:-1,0:2] = 11.0
print(obj.position)
print("counter: ", obj.position.getCounter())


obj.position[1] = 4.0
print(obj.position)
print("counter: ", obj.position.getCounter())

obj.showColor = [0.1, 0.1, 0.1, 0.1]
print("1: ", obj.showColor)
print("counter: ", obj.showColor.getCounter())

obj.showColor[0] = 2.0
print("2: ", obj.showColor)
print("counter: ", obj.showColor.getCounter())

obj.showColor[2:] = 4.0
print("3: ", obj.showColor)
print("counter: ", obj.showColor.getCounter())


#obj.position[0] = [1,2,3]
#obj.position[1] = [2,3,4]
#obj.position[1:2] = [2,3,4]

#obj.position[0,:] = 1
#obj.position[1,:] = 2
#obj.position[1:2,:] = 2


