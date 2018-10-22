# coding: utf8

import sys
import os
import numpy
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

SofaRuntime.importPlugin("SofaAllCommonComponents")

import numpy
rawcpy = numpy.zeros((1000000,3), dtype=numpy.float64)
slowcpy = numpy.zeros((1000000,3), dtype=numpy.float32)
aList = rawcpy.tolist()

root = Sofa.Node("root")
obj = root.createObject("MechanicalObject", name="test", position=aList)

obj.position = aList
obj.position = rawcpy
obj.position = slowcpy

p = obj.position

d = rawcpy.copy()

import timeit
it = 1000

def pattern1SofaPython():
        l = obj.position.tolist()
        for i in range(0, len(l)):
                l[i]=[l[i][0]+1.0,l[i][0]+2.0,l[i][0]+3.0]
        obj.position = l

l = [1.0,2.0,3.0]        
v = numpy.zeros((1,3))

def pattern1SofaPython3():
        global l
        obj.position += l

def pattern1bisSofaPython3():
        global v
        obj.position += v


def pattern2SofaPython():
        global aList
        l = obj.position.tolist()
        for i in range(0, len(l)):
                l[i]=[l[i][0]+aList[i][0],l[i][1]+aList[i][1],l[i][2]+aList[i][2]]
        obj.position = l

def pattern2SofaPython3():
        global rawcpy
        obj.position += rawcpy

        
print(timeit.timeit("obj.position.tolist()", number=it, globals=globals()))
print(timeit.timeit("obj.position", number=it, globals=globals()))
#print(timeit.timeit("obj.position[0,0]", number=it, globals=globals()))
it = 10

print("pattern 1")
print(timeit.timeit("pattern1SofaPython()", number=it, globals=globals()))
print(timeit.timeit("pattern1SofaPython3()", number=it, globals=globals()))
print(timeit.timeit("pattern1bisSofaPython3()", number=it, globals=globals()))

print("pattern 2")
print(timeit.timeit("pattern2SofaPython()", number=it, globals=globals()))
print(timeit.timeit("pattern2SofaPython3()", number=it, globals=globals()))


print("setattr")
print(timeit.timeit("obj.position=aList", number=it, globals=globals()))
print(timeit.timeit("obj.position=rawcpy", number=it, globals=globals()))
print(timeit.timeit("obj.position=slowcpy", number=it, globals=globals()))
print("lastmile")
print(timeit.timeit("d = obj.position.copy()", number=it, globals=globals()))
print(timeit.timeit("d1 = rawcpy.copy()", number=it, globals=globals()))
print(timeit.timeit("d2 = slowcpy.copy()", number=it, globals=globals()))

