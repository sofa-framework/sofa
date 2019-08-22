
import xml.etree.ElementTree as ET


def requires(node, *names):

    for name in names:
        node.createObject( 'RequiredPlugin',
                           name = name,
                           pluginName = name )

def display_flags(node, **kwargs):

    items = [k + x for k, v in kwargs.iteritems() for x in v.split()]
    node.createObject('VisualStyle', displayFlags = ' '.join(items))

        

def xml_insert(sofanode, xmlnode):
    '''load xml node under sofa node'''
    
    if xmlnode.tag == 'Node':

        name = xmlnode.attrib.get('name', '')
        sofachild = sofanode.createChild(name)
        
        for xmlchild in xmlnode:
            xml_insert(sofachild, xmlchild)

        return sofachild
    else:
        return sofanode.createObject(xmlnode.tag, **xmlnode.attrib)
        
def xml_load(filename):
    
    return ET.parse('{0}/{1}'.format(path(), filename)).getroot()


import sys
import Sofa

class Script(Sofa.PythonScriptController):

    def __new__(cls, node):

        module = sys.modules[cls.__module__]
        name = cls.__name__
        file = module.__file__

        
        res = node.createObject('PythonScriptController',
                                filename=file,
                                classname=name)
        
        # update class object since createObject reload the script
        cls = getattr(module, name)
        
        return cls.__instance__
    
    def onLoaded(self, node):
        type(self).__instance__ = self
        


def contacts(node, **kwargs):
    node.createObject('DefaultPipeline')
    node.createObject('BruteForceDetection')
    node.createObject('NewProximityIntersection',
                      alarmDistance = kwargs.get('alarm_dist', 0.02),
                      contactDistance = kwargs.get('contact_dist', 0.01))
    node.createObject('DefaultContactManager',
                      response = kwargs.get('response',
                                            'FrictionCompliantContact'))

import sys
import os
def path():
    return os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
