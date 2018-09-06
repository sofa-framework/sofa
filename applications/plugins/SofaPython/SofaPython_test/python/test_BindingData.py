# -*- coding: utf-8 -*-
import Sofa
from SofaTest import *
import array
import numpy as np
def tolist(o):
    if o.ndim == 0:
        return o.value
    elif o.ndim == 1:
        tmp=[]
        for i in range(o.shape[0]):
            tmp.append(o[i])

        return tmp
    elif o.ndim == 2:
        tmp=[]
        for i in range(o.shape[0]):
            tmp2=[]
            for j in range(o.shape[1]):
                tmp2.append(o[i,j])
            tmp.append(tmp2)
        return tmp
    else:
        return None

def OnDiagonal(obj):
    if obj.ndim != 2:
        print("ZUT")

    for i in xrange(0, obj.shape[0]):
        for j in xrange(0, obj.shape[1]):
            yield (i,j)

def OnEach(obj):
    if obj.ndim==1:
        for i in xrange(0, obj.shape[0]):
            yield i
    else:
        for i in xrange(0, obj.shape[0]):
            for j in xrange(0, obj.shape[1]):
                yield (i,j)

def OnEven(obj):
    for i in xrange(0, obj.shape[0], 1):
        for j in xrange(0, obj.shape[1], 2):
            yield (i,j)

def accessor(rootNode):
    print("======================BEGIN TEST==============")
    o = rootNode.createObject("MechanicalObject", template="Rigid", position=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    o.addNewData("test", "", "docstring", "t", [[10,2,4,6]])

    for name in ["position", "showColor", "listening", "name", "test", "translation"]:
        p = o.getData(name)
        print("Testing => "+p.name)
        print("   type: " +p.type)

        print("   ndim: " + str(p.ndim) )
        print("  shape: " + str(p.shape) + " size("+str(p.ndim)+")" )
        print(" values: "+str(tolist(p)))


    print("=== 1D structure ===")
    c = o.getData("showColor")
    print("color is: "+repr(c))

    c[OnEach] = 2.0
    print(" onEach: "+repr(c))

    c[0:2] = 1.0
    print("  slice: "+repr(c))


    print("=== 2D structure ===")
    p = o.getData("position")
    print("initial: "+repr(p))
    p[0] = [4.0,4.0,4.0,4.0,4.0,4.0,4.0]
    p[1] = [5.0,5.0,5.0,5.0,5.0,5.0,5.0]
    print("axis   : "+repr(p))

    p[0,0] = 0.0
    p[0,1] = 0.0
    p[1,0] = 1.0
    p[1,1] = 1.0
    print("single : "+repr(p))

    p[OnEach] = 3.0
    print("OnEach : "+repr(p))

    p[OnEven] = 2.0
    print("OnEven : "+repr(p))

    p[:,0:4] = 0.0
    p[:,2:] = 2.0
    print("Slices : "+repr(p))

def createScene(rootNode):
    rootNode.addNewData("aField", "TestField", "help message", "float", 1.0)
    field = rootNode.getData("aField")
    ASSERT_NEQ(field, None)

    ### Check isPersistant/setPersistant
    ASSERT_TRUE( field.isPersistant() )
    field.setPersistant(False)
    ASSERT_FALSE( field.isPersistant() )

    ### Check isSet/unset
    ASSERT_TRUE( field.isSet() )
    field.unset()
    ASSERT_FALSE( field.isSet() )

    ### Check the hasParent/parentPath
    ASSERT_EQ( field.hasParent(), field.getParentPath() == True )

    ### SetValueString
    t = field.getCounter()
    field.setValueString("2.0")
    ASSERT_NEQ(field.getCounter(), t)

    ### Different ways to get values.
    ASSERT_GT(field.getValue(0), 1.5)
    ASSERT_GT(field.value, 1.5)

    t = field.getCounter()
    field.value = 3.0
    ASSERT_NEQ(field.getCounter(), t)
    ASSERT_GT(field.getValue(0), 2.5)
    ASSERT_GT(field.value, 2.5)

    ### What happens if we do something wrong ?
    t = field.getCounter()
    field.setValueString("7.0 8.0 9.0")
    ASSERT_NEQ(field.getCounter(), t)

    #TODO(dmarchal 2017-11-17) well returning zero size and allowing a zero index looks not really
    #good API design.
    ASSERT_EQ(field.getSize(), 0)
    ASSERT_GT(field.getValue(0), 6.0)

    ASSERT_EQ(type(field.isRequired()), bool)
    ASSERT_EQ(type(field.isReadOnly()), bool)
    ASSERT_EQ(type(field.getHelp()), str)

    accessor(rootNode)
