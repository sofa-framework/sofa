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


rawcpy = numpy.zeros((10000,3), dtype=numpy.float64)
aList = rawcpy.tolist()

root = Sofa.Node("root")
obj = root.createObject("MechanicalObject", name="test", position=[])
obj.position = aList
d = obj.position

print("bench: ", obj.getData("position").getCounter() )
print("simple get: ", timeit.timeit("obj.listening", globals=globals(), number=10000))
print("complex data: ", timeit.timeit("obj.position", globals=globals(), number=10000))
print("old sofa: ", timeit.timeit("oldSofa(obj)", globals=globals(), number=10000))

print("counter:", obj.getData("position").getCounter())

print(timeit.timeit("obj.position = obj.position+0.5", globals=globals(), number=10000))
print("counter:", obj.getData("position").getCounter())

print(timeit.timeit("obj.position = d+0.5", globals=globals(), number=10000))
print("counter:", obj.getData("position").getCounter())

print(timeit.timeit("obj.position += 0.5", globals=globals(), number=10000))
print("counter:", obj.getData("position").getCounter())

d=[obj.position]
print(timeit.timeit("d[0] += 0.5", globals=globals(), number=10000))
print("counter (broken):", obj.getData("position").getCounter())


with obj.WriteAccessor("position") as p:
        d = [p]
        print(timeit.timeit("d[0] += 0.5", globals=globals(), number=10000))
        print("counter:", obj.getData("position").getCounter())

print("counter:", obj.getData("position").getCounter())
               
print(str(obj.position))

