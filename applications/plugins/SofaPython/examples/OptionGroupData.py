import Sofa
import sys

def createSceneAndController(node):

    defaultContactManager = node.createObject( 'DefaultContactManager', response="disabled" )

    og = defaultContactManager.findData('response')
    print og
    print 'getSize()',og.getSize()
    print "getCurrentItem()", og.getCurrentItem(), "getCurrentId()", og.getCurrentId()
    print "currentItem", og.currentItem, "currentId", og.currentId
    print "getItem(1)", og.getItem(1)

    print 'setCurrentItem("default")'
    og.setCurrentItem("default")
    print "currentItem", og.currentItem, "currentId", og.currentId

    print 'setCurrentId(0)'
    og.setCurrentId(0)
    print "currentItem", og.currentItem, "currentId", og.currentId

    print 'currentItem = "default"'
    og.currentItem = "default"
    print "currentItem", og.currentItem, "currentId", og.currentId

    print 'currentId=0'
    og.currentId=0
    print "currentItem", og.currentItem, "currentId", og.currentId


    print defaultContactManager.response
    print "currentItem", defaultContactManager.response.currentItem, "currentId", defaultContactManager.response.currentId



    sys.stdout.flush()
