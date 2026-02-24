import importlib  
Bar16_spring_rk4 = importlib.import_module("Bar16-spring-rk4")

def createScene(root_node):
    root_node.addObject('RequiredPlugin', name='SofaCUDA')
    Bar16_spring_rk4.createScene(root_node, "CudaVec3f")
