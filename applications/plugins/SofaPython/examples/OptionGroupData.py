import Sofa
import sys

def createSceneAndController(node):

    defaultContactManager = node.createObject( 'DefaultContactManager', response="disabled" )

    og = defaultContactManager.findData('response')
    print og
    print 'getSize()',og.getSize()
    print "getSelectedItem()", og.getSelectedItem(), "getSelectedId()", og.getSelectedId()
    print "selectedItem", og.selectedItem, "selectedId", og.selectedId
    print "getItem(1)", og.getItem(1)

    print 'setSelectedItem("default")'
    og.setSelectedItem("default")
    print "selectedItem", og.selectedItem, "selectedId", og.selectedId

    print 'setSelectedId(0)'
    og.setSelectedId(0)
    print "selectedItem", og.selectedItem, "selectedId", og.selectedId

    print 'selectedItem = "default"'
    og.selectedItem = "default"
    print "selectedItem", og.selectedItem, "selectedId", og.selectedId

    print 'selectedId=0'
    og.selectedId=0
    print "selectedItem", og.selectedItem, "selectedId", og.selectedId


    print defaultContactManager.response, defaultContactManager.response.selectedItem, defaultContactManager.response.selectedId
    print defaultContactManager.response.getValueString()



    sys.stdout.flush()
