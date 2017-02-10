import Sofa
import sys

def createScene(node):

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


    print defaultContactManager.response, defaultContactManager.response.selectedItem, defaultContactManager.response.selectedId, defaultContactManager.response.getValueString()



    print '\n=== Copy a Sofa.OptionsGroupData in another one ==='

    defaultContactManager2 = node.createObject( 'DefaultContactManager', response="disabled", name="2" )

    print "target:", defaultContactManager.response.selectedItem, defaultContactManager.response.selectedId
    print "source before copy:", defaultContactManager2.response.selectedItem, defaultContactManager2.response.selectedId

    defaultContactManager2.response = og

    print "source after copy:", defaultContactManager2.response.selectedItem, defaultContactManager2.response.selectedId


    sys.stdout.flush()
