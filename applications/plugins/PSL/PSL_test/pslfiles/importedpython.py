PSLExport = ["Template1", "MyTemplate"]

def Template1(templateRoot):
    Sofa.msg_info(templateRoot, "This have been created from a python template")

def MyTemplate(templateRoot, numchild=10):
    Sofa.msg_info(templateRoot, "This have been created from a python template")
    for i in range(0, numchild):
        templateRoot.createChild("Something"+str(i))
