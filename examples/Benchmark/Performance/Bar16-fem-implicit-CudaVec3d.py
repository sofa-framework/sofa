import importlib  
bar16_fem_implicit = importlib.import_module("Bar16-fem-implicit")

def createScene(root_node):
    root_node.addObject('RequiredPlugin', pluginName='SofaCUDA')
    bar16_fem_implicit.internalCreateScene(root_node, "CudaVec3d")
