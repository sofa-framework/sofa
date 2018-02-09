import pprint
import psl
from psl.dsl import *

@psltemplate
def Template1(templateRoot, name):
    Sofa.msg_info(templateRoot, "This have been created from a python template")

@psltemplate
def MyTemplate(templateRoot, name, numchild=10):
    Sofa.msg_info(templateRoot, "This have been created from a python template")
    for i in range(0, numchild):
        templateRoot.createChild("Something"+str(i))
