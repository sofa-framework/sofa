"""
Original Contributors:
    - jeremie.allard@insimo.fr (InSimo)
    - remi.bessard_duparc@inria.fr (Inria)
    - frederick.roy@inria.fr (Inria)
"""

import math

import Sofa


# 3 distinct torus appearances: (collision_color, visual_material)
DARK_BROWN = (
    "0.18039215686275 0.12156862745098 0.086274509803922",
    "mat Ambient 1 0.036078431372549 0.024313725490196 0.017254901960784 1.0"
    " Diffuse 1 0.18039215686275 0.12156862745098 0.086274509803922 1.0"
    " Specular 1 1.0 1.0 1.0 1.0"
    " Emissive 0 0.18039215686275 0.12156862745098 0.086274509803922 1.0"
    " Shininess 1 40",
)
TAN = (
    "0.7921568627451 0.50196078431373 0.2156862745098",
    "mat Ambient 1 0.15843137254902 0.10039215686275 0.043137254901961 1.0"
    " Diffuse 1 0.7921568627451 0.50196078431373 0.2156862745098 1.0"
    " Specular 1 1.0 1.0 1.0 1.0"
    " Emissive 0 0.7921568627451 0.50196078431373 0.2156862745098 1.0"
    " Shininess 1 40",
)
IVORY = (
    "0.94117647058824 0.93725490196078 0.89411764705882",
    "mat Ambient 1 0.18823529411765 0.18745098039216 0.17882352941176 1.0"
    " Diffuse 1 0.94117647058824 0.93725490196078 0.89411764705882 1.0"
    " Specular 1 1.0 1.0 1.0 1.0"
    " Emissive 0 0.94117647058824 0.93725490196078 0.89411764705882 1.0"
    " Shininess 1 40",
)

# Per-torus data: (activated, translation, rotation, appearance)
TORUS_DATA = [
    (True,  "-17.800790396426 30 18.046774108916",  "35.624275834125 79.418789031645 -15.634653477759",   DARK_BROWN),
    (True,  "31.203746381776 46 -3.9584052767411",  "-81.407505530588 -41.699467018107 65.027568971285",  DARK_BROWN),
    (True,  "-22.646083078648 62 -29.076289417723", "-48.524586576281 9.6813763210929 49.716395474838",    TAN),
    (True,  "-0.36212878318602 78 39.681506827325", "58.536139097314 -42.029633536949 -32.440462048371",  TAN),
    (True,  "23.392304286078 94 -30.079806740433",  "39.709873269177 15.292576442143 7.9574386952247",    TAN),
    (False, "-39.61300034058 110 0.86055988485951",  "45.193954620135 -42.616531463627 77.409719260135",   TAN),
    (False, "17.130322557469 126 17.800171588454",   "72.186179143463 59.053434119119 88.31836399544",     IVORY),
    (False, "-10.390362092476 142 -33.168565888502", "-89.649008726538 -67.300238701189 -9.8223450965352", IVORY),
    (False, "-26.521328197104 158 28.966597611535",  "14.924939379527 81.74878913525 -44.199652361777",    IVORY),
    (False, "35.093469803731 174 -3.1191041148822",  "-5.1181413303679 82.199841808621 -51.436056066973",  IVORY),
]

N_CYLINDERS = 16
CYLINDER_RADIUS = 100.0
CYLINDER_HEIGHT = 61.75


def createScene(root):
    root.gravity = [0, -100, 0]
    root.dt = 0.01

    root.addObject('RequiredPlugin', pluginName='Sofa.Component.AnimationLoop')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Collision.Detection.Algorithm')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Collision.Detection.Intersection')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Collision.Geometry')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Collision.Response.Contact')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Constraint.Lagrangian.Correction')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Constraint.Lagrangian.Solver')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Engine.Generate')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Engine.Transform')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.IO.Mesh')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.LinearSolver.Direct')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Mapping.Linear')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Mass')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.ODESolver.Backward')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.SolidMechanics.FEM.Elastic')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.StateContainer')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Topology.Container.Dynamic')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Topology.Container.Grid')
    root.addObject('RequiredPlugin', pluginName='Sofa.Component.Visual')
    root.addObject('RequiredPlugin', pluginName='Sofa.GL.Component.Rendering3D')

    root.addObject('VisualStyle', displayFlags='showVisual showCollisionModels hideBehaviorModels showForceFields hideInteractionForceFields hideWireframe')

    root.addObject('FreeMotionAnimationLoop', name='animationLoop',
                    solveVelocityConstraintFirst=True,
                    parallelCollisionDetectionAndFreeMotion=True,
                    parallelODESolving=True)
    root.addObject('BlockGaussSeidelConstraintSolver', name='constraintSolver',
                    maxIterations=100, tolerance=1.0e-9, multithreading=True)
    root.addObject('CollisionPipeline', name='Pipeline')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='ContactManager',
                    response='FrictionContactConstraint', responseParams='mu=0')
    root.addObject('NewProximityIntersection', name='Intersection',
                    alarmDistance=2.5, contactDistance=0.01)

    # ---- Mesh node: shared loaders and engines ----
    mesh = root.addChild('Mesh')
    mesh.addObject('MeshOBJLoader', name='MeshTorus1V', filename='mesh/torus1.obj',
                    scale3d=[10, 10, 10], triangulate=True)
    mesh.addObject('MeshOBJLoader', name='MeshTorus1C', filename='torus1-16x8.obj',
                    scale3d=[10, 10, 10], triangulate=True)
    mesh.addObject('MeshOBJLoader', name='MeshTorus1D05', filename='mesh/torus1-dilated-05.obj',
                    scale3d=[10, 10, 10], triangulate=True)
    mesh.addObject('MeshTetraStuffing', name='stuffing', snapPoints=False, splitTetras=False,
                    draw=False, size=13.859131548844, alphaLong=0.4, alphaShort=0.4,
                    inputPoints='@MeshTorus1D05.position',
                    inputTriangles='@MeshTorus1D05.triangles')
    mesh.addObject('TransformEngine', name='TorusVVel',
                    input_position='@stuffing.outputPoints',
                    translation=[0, -100, 0], scale=[0, 0, 0])
    mesh.addObject('MeshOBJLoader', name='MeshCylinder', filename='mesh/cylinder.obj',
                    scale3d=[6.5, 6.5, 6.5], triangulate=True)

    # ---- Simulation node: deformable tori ----
    simulation = root.addChild('Simulation')

    for i, (activated, translation, rotation, (col_color, vis_material)) in enumerate(TORUS_DATA):
        torus = simulation.addChild(f'Torus{i}')
        torus.activated = activated

        torus.addObject('EulerImplicitSolver', name=f'ODE{i}',
                        rayleighMass=0.1, rayleighStiffness=0.1)
        torus.addObject('SparseLDLSolver', name=f'Linear{i}',
                        parallelInverseProduct=True,
                        template='CompressedRowSparseMatrixMat3x3')
        torus.addObject('TransformEngine', name=f'Torus{i}VXForm',
                        input_position='@/Mesh/stuffing.outputPoints',
                        translation=translation, rotation=rotation)
        torus.addObject('TetrahedronSetTopologyContainer', name=f'Torus{i}Topo',
                        position='@/Mesh/stuffing.outputPoints',
                        tetrahedra='@/Mesh/stuffing.outputTetrahedra')
        torus.addObject('TetrahedronSetTopologyModifier', name=f'Torus{i}Mod')
        torus.addObject('MechanicalObject', name=f'Torus{i}State', template='Vec3',
                        velocity='@/Mesh/TorusVVel.output_position',
                        rest_position='@/Mesh/stuffing.outputPoints',
                        position=f'@Torus{i}VXForm.output_position')
        torus.addObject('TetrahedronSetGeometryAlgorithms', name=f'Torus{i}Algo')
        torus.addObject('UniformMass', totalMass=20)
        torus.addObject('TetrahedronFEMForceField', name='FEM',
                        youngModulus=50, poissonRatio=0.45)
        torus.addObject('GenericConstraintCorrection', name=f'Torus{i}CC',
                        linearSolver=f'@Linear{i}')

        surface = torus.addChild('Surface')
        surface.addObject('TriangleSetTopologyContainer',
                          position='@/Mesh/MeshTorus1C.position',
                          triangles='@/Mesh/MeshTorus1C.triangles')
        surface.addObject('MechanicalObject', template='Vec3')
        surface.addObject('BarycentricMapping', useRestPosition=True,
                          mapForces=False, mapMasses=False, mapConstraints=True)
        surface.addObject('TriangleCollisionModel', name=f'Torus{i}CMT',
                          contactDistance=0.2, color=col_color)

        visual = torus.addChild('Visual')
        visual.addObject('OglModel', name=f'Torus{i}VM',
                         position='@/Mesh/MeshTorus1V.position',
                         restPosition='@/Mesh/MeshTorus1V.position',
                         triangles='@/Mesh/MeshTorus1V.triangles',
                         material=vis_material, handleDynamicTopology=False)
        visual.addObject('BarycentricMapping', useRestPosition=True)

    # ---- Obstacles node: cylindrical posts and floor ----
    obstacles = root.addChild('Obstacles')

    for i in range(N_CYLINDERS):
        angle = i * 2.0 * math.pi / N_CYLINDERS
        x = CYLINDER_RADIUS * math.cos(angle)
        z = CYLINDER_RADIUS * math.sin(angle)

        cyl = obstacles.addChild(f'CylA_{i}')
        cyl.addObject('EdgeSetTopologyContainer', name=f'Cyl{i}LineTopo',
                       position=f'{x} 0 {z}  {x} {CYLINDER_HEIGHT} {z}',
                       edges='0 1')
        cyl.addObject('MechanicalObject', name=f'Cyl{i}LineState', template='Vec3')
        cyl.addObject('EdgeSetGeometryAlgorithms', name=f'Cyl{i}LineAlgo')
        cyl.addObject('LineCollisionModel', name=f'Cyl{i}CML',
                       contactDistance=6.5, moving=0, simulated=0)

        cyl_visual = cyl.addChild('Visual')
        cyl_visual.addObject('TransformEngine', name=f'Cyl{i}SurfXForm',
                             input_position='@/Mesh/MeshCylinder.position',
                             translation=[x, 0, z])
        cyl_visual.addObject('OglModel', name=f'Cyl{i}VM',
                             position=f'@Cyl{i}SurfXForm.output_position',
                             triangles='@/Mesh/MeshCylinder.triangles',
                             color='0.23 0.25 0.42')

    floor = obstacles.addChild('Floor')
    floor.addObject('TriangleSetTopologyContainer', name='FloorTopo',
                    position='-1000 -5 -1000  1000 -5 -1000  1000 -5 1000  -1000 -5 1000',
                    triangles='0 2 1  3 2 0')
    floor.addObject('MechanicalObject', template='Vec3')
    floor.addObject('TriangleCollisionModel', name='FloorCM',
                    contactDistance=5, moving=0, simulated=0)

    floor_visu = floor.addChild('Visu')
    floor_visu.addObject('RegularGridTopology', name='FloorTopo',
                         nx=20, ny=1, nz=20,
                         min=[1000, 0, -1000], max=[-1000, 0, 1000])
    floor_visu.addObject('OglModel', name='Visual', color='0.52 0.46 0.4',
                         material='floor Ambient 1 0.01 0.01 0.01 0.0 Diffuse 1 0.52 0.46 0.4 1.0 Specular 0 1.0 1.0 1.0 1.0 Emissive 0 0.05 0.05 0.05 0.0 Shininess 0 20')

    return root
