import Sofa
import sys

def createScene(node):

    node.createObject("MechanicalObject",position="0 0 0   1 0 0  2 0 0")
    sff = node.createObject("SpringForceField", spring="0 1 10 0.1 1    1 2 10 0.1 1")


    print type(sff.spring), sff.spring
    print type(sff.spring.value), sff.spring.value
    print sff.spring.value[0][0].Ks

    ls = Sofa.LinearSpring(0, 1, 100., 100., 100.)

    sff.spring.value = ls
    print len(sff.spring.value),sff.spring.value[0][0].Ks
    sff.spring.value = [ls,ls]
    print len(sff.spring.value),sff.spring.value[1][0].Ks


    print 'len',len(sff.spring)

    print sff.spring.__getitem__(0).Ks

    ls.Ks = 99999
    sff.spring[1] = ls
    print sff.spring[0].Ks, sff.spring[1].Ks
    sff.spring[0] = sff.spring[1]
    print sff.spring[0].Ks, sff.spring[1].Ks


    sys.stdout.flush()