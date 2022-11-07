import Sofa


#Choose in your script to activate or not the GUI
USE_GUI = True


def main():
    import SofaRuntime
    import Sofa.Gui

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()

def createScene(root):
    
    root.gravity=[0, 0, 0]
    root.dt=0.05
    root.showBoundingTree = 0
    root.addObject('RequiredPlugin', name="plug1", pluginName="Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact")
    root.addObject('RequiredPlugin', name="plug2", pluginName="Sofa.Component.Constraint.Projective Sofa.Component.LinearSolver.Iterative Sofa.Component.ODESolver.Backward")
    root.addObject('RequiredPlugin', name="plug3", pluginName="Sofa.Component.Engine.Select Sofa.Component.IO.Mesh Sofa.Component.Mass ")
    root.addObject('RequiredPlugin', name="plug4", pluginName="Sofa.Component.Mapping.Linear Sofa.Component.Mapping.NonLinear Sofa.Component.SolidMechanics.FEM.Elastic Sofa.Component.StateContainer")
    root.addObject('RequiredPlugin', name="plug5", pluginName="Sofa.Component.Topology.Container.Dynamic Sofa.Component.Topology.Mapping")
    root.addObject('RequiredPlugin', name="plug6", pluginName="Sofa.Component.Visual Sofa.GL.Component.Rendering3D")
    root.addObject('RequiredPlugin', name="plug7", pluginName="Sofa.Component.Collision.Detection.Algorithm SofaCarving")

    root.addObject('VisualStyle',displayFlags="")
    root.addObject('DefaultAnimationLoop')
    root.addObject('DefaultPipeline',verbose="0")
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('DefaultContactManager',response="PenalityContactForceField")
    root.addObject('MinProximityIntersection',name="Proximity",alarmDistance="0.08",contactDistance="0.05",useSurfaceNormals="false")
    root.addObject('CarvingManager',active="true",carvingDistance="-0.01")
  
 
  
    TT = root.addChild('TT')
    
    TT.addObject('EulerImplicitSolver',name="cg_odesolver",printLog="false",rayleighStiffness="0.1",rayleighMass="0.1")
    TT.addObject('CGLinearSolver',iterations="25",name="linear solver",tolerance="1.0e-9",threshold="1.0e-9")
    TT.addObject('MeshGmshLoader',filename="mesh/liver.msh",name="loader",scale="1")   
   
    TT.addObject('MechanicalObject',template="Vec3d",src="@loader",name="Volume")
    
    TT.addObject('TetrahedronSetTopologyContainer', name="topo", src="@loader")
    TT.addObject('TetrahedronSetTopologyModifier', name="topoMod")
    TT.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="GeomAlgo")
    
    TT.addObject('DiagonalMass',massDensity="0.5")
    TT.addObject('FixedConstraint',indices="1 3 50")
    TT.addObject('TetrahedralCorotationalFEMForceField',name="CFEM",youngModulus="160",poissonRatio="0.3",method="large")
    
    T=TT.addChild('T')    
    T.addObject('TriangleSetTopologyContainer', name="Container")
    T.addObject('TriangleSetTopologyModifier', name="Modifier")
    T.addObject('TriangleSetGeometryAlgorithms',name="GeomAlgo",template="Vec3d")
    T.addObject('Tetra2TriangleTopologicalMapping', input="@../topo", output="@Container")
    
    T.addObject('TriangleCollisionModel', tags="CarvingSurface")

    Visu = T.addChild('Visu')
    Visu.addObject('OglModel',name="Visual",material="Default Diffuse 1 0 1 0 1 Ambient 0 1 1 1 1 Specular 1 1 1 0 1 Emissive 0 1 1 0 1 Shininess 1 100")
    Visu.addObject('IdentityMapping',input="@Volume",output="@Visual")


    Instrument = root.addChild('Instrument')
    Instrument.addObject('EulerImplicitSolver',name="cg_odesolver",printLog="false")
    Instrument.addObject('CGLinearSolver',iterations="25",name="linear solver",tolerance="1.0e-9",threshold="1.0e-9")
    Instrument.addObject('MechanicalObject',template="Rigid3d",name="instrumentState",tags="Omni",rotation="90 45 0",translation="0 0 1")
    Instrument.addObject('UniformMass',template="Rigid3d",name="mass",totalMass="5")
    visuIns = Instrument.addChild('visuIns')
    visuIns.addObject('MeshOBJLoader',name="meshLoader_0",filename="mesh/dental_instrument_light.obj",scale3d="1 1 1",translation="-0.412256 -0.067639 3.35",rotation="180 0 150",handleSeams="1")
    visuIns.addObject('OglModel',template="Vec3d",name="InstrumentVisualModel",src="@meshLoader_0",material="Default Diffuse 1 1 0.2 0.2 1 Ambient 1 0.2 0.04 0.04 1 Specular 0 1 0.2 0.2 1 Emissive 0 1 0.2 0.2 1 Shininess 0 45")
    visuIns.addObject('RigidMapping',template="Rigid3d,Vec3d",name="MM->VM mapping",input="@instrumentState",output="@InstrumentVisualModel")

    colIns = Instrument.addChild('colIns')
    colIns.addObject('MechanicalObject',template="Vec3d",name="Particle",position="-0.2 -0.2 -0.2")
    colIns.addObject('SphereCollisionModel',name="ParticleModel",radius="0.2",tags="CarvingTool")
    #colIns.addObject('SphereCollisionModel',name="ParticleModel",radius="0.2")
    colIns.addObject('RigidMapping',template="Rigid3d,Vec3d",name="MM->CM mapping",input="@instrumentState",output="@Particle")

    return root


#Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()
