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

it=1000

root = Sofa.Node("rootNode")
c = []
for i in range(0,1000):
        newchild = root.createChild("Damien"+str(i))
        c.append(newchild)
        
parented = root.createChild("parented")
for child in root.children:
        child.addChild(parented)        


def oldIt():
        c = []
        for e in root.__old_getChildren():
                c.append(e)         

def newIt():
        c = []
        for e in root.children:
                c.append(e)         

def oldRange():
        c = []
        for e in range(len(root.__old_getChildren())):
                c.append(root.__old_getChild(e))         

def newRange():
        c = []
        for e in range(len(root.children)):
                c.append(root.children[e])         

def oldRangeF():
        c = []
        for e in range(len(root.children)):
                c.append(root.__old_getChild(e))         

def newRangeF():
        c = []
        d = root.children
        for e in range(len(d)):
                c.append(d[e])         



code = ["parented.parents",
        "parented.parents[0]",
        "list(parented.parents)",
        "len(parented.parents)",
        "root.children",
        "root.children[50]",
        "root.__old_getChild(50)",
        "root.children[50]",
        "root.__old_getChild(50)",
        "len(root.children)",
        "len(root.__old_getChildren())",
        "list(root.children)",
        "root.__old_getChildren()",
        "oldIt()",
        "newIt()",
        "oldRange()",
        "newRange()",
        "oldRangeF()",
        "newRangeF()"]         

for c in code:
        print(c, timeit.timeit(c, number=it, globals=globals()))

