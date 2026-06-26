import importlib  
Bar16_spring_rk4 = importlib.import_module("Bar16-spring-rk4")

def createScene(root_node):
    Bar16_spring_rk4.internalCreateScene(root_node, "Vec3f")
