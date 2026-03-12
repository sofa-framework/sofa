import importlib  
bar16_fem_implicit = importlib.import_module("Bar16-fem-implicit")

def createScene(root_node):
    bar16_fem_implicit.createScene(root_node, "Vec3d")