/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/component/sceneutility/InfoComponent.h>
using sofa::component::sceneutility::InfoComponent;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/simpleapi/SimpleApi.h>

#include <thread>

class ParallelScenesTest : public virtual sofa::testing::BaseTest
{
public:

    ParallelScenesTest()
    {
    }
    
    void executeInParallel(const char* sceneStr, const std::size_t nbScenes, const std::size_t nbSteps)
    {
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_MSG_NOEMIT(Warning);
        
        std::vector<sofa::simulation::NodeSPtr> groots;
        groots.resize(nbScenes);
        std::size_t index = 0;
        for (auto& groot : groots)
        {
            groot = SceneLoaderXML::loadFromMemory("testscene", sceneStr);
            ASSERT_TRUE(groot);

            sofa::simulation::node::initRoot(groot.get());
            groot->setAnimate(true);
            
            groot->setName("Scene " + std::to_string(index));
            index++;
        }
        
        auto simuLambda = [&](auto groot)
            {
                std::size_t counter = 0;
                msg_info(groot->getName()) << "start";
                while (counter < nbSteps)
                {
                    sofa::simulation::node::animate(groot.get());
                    counter++;
                }
                msg_info(groot->getName()) << "end";
                
                EXPECT_TRUE(counter == nbSteps);
            };
        
        std::vector<std::thread> threads;
        for(auto& groot : groots)
        {
            threads.emplace_back(simuLambda, groot);
        }

        for(auto& t : threads)
        {
            t.join();
        }
        
        for (auto& groot : groots)
        {
            ASSERT_TRUE(groot);
            sofa::simulation::node::unload(groot);
        }
    }

    void testParallelLiver(const std::size_t nbScenes, const std::size_t nbSteps)
    {
        const std::string sceneStr = R"(
        <?xml version="1.0" ?>
        <Node name="lroot" gravity="0 -9.81 0" dt="0.02">
            <RequiredPlugin name="Sofa.Component.Collision.Detection.Algorithm"/> <!-- Needed to use components [BVHNarrowPhase BruteForceBroadPhase CollisionPipeline] -->
            <RequiredPlugin name="Sofa.Component.Collision.Detection.Intersection"/> <!-- Needed to use components [DiscreteIntersection] -->
            <RequiredPlugin name="Sofa.Component.Collision.Geometry"/> <!-- Needed to use components [SphereCollisionModel] -->
            <RequiredPlugin name="Sofa.Component.Collision.Response.Contact"/> <!-- Needed to use components [CollisionResponse] -->
            <RequiredPlugin name="Sofa.Component.Constraint.Projective"/> <!-- Needed to use components [FixedProjectiveConstraint] -->
            <RequiredPlugin name="Sofa.Component.IO.Mesh"/> <!-- Needed to use components [MeshGmshLoader MeshOBJLoader SphereLoader] -->
            <RequiredPlugin name="Sofa.Component.LinearSolver.Iterative"/> <!-- Needed to use components [CGLinearSolver] -->
            <RequiredPlugin name="Sofa.Component.Mapping.Linear"/> <!-- Needed to use components [BarycentricMapping] -->
            <RequiredPlugin name="Sofa.Component.Mass"/> <!-- Needed to use components [DiagonalMass] -->
            <RequiredPlugin name="Sofa.Component.ODESolver.Backward"/> <!-- Needed to use components [EulerImplicitSolver] -->
            <RequiredPlugin name="Sofa.Component.SolidMechanics.FEM.Elastic"/> <!-- Needed to use components [TetrahedralCorotationalFEMForceField] -->
            <RequiredPlugin name="Sofa.Component.StateContainer"/> <!-- Needed to use components [MechanicalObject] -->
            <RequiredPlugin name="Sofa.Component.Topology.Container.Dynamic"/> <!-- Needed to use components [TetrahedronSetGeometryAlgorithms TetrahedronSetTopologyContainer] -->
            <RequiredPlugin name="Sofa.GL.Component.Rendering3D"/> <!-- Needed to use components [OglModel] -->

            <CollisionPipeline name="CollisionPipeline" verbose="0" />
            <DefaultAnimationLoop/>
            <BruteForceBroadPhase/>
            <BVHNarrowPhase/>
            <CollisionResponse name="collision response" response="PenalityContactForceField" />
            <DiscreteIntersection/>

            <MeshOBJLoader name="LiverSurface" filename="mesh/liver-smooth.obj" />

            <Node name="Liver" gravity="0 -9.81 0">
                <EulerImplicitSolver name="cg_odesolver"   rayleighStiffness="0.1" rayleighMass="0.1" />
                <CGLinearSolver name="linear solver" iterations="25" tolerance="1e-09" threshold="1e-09" />
                <MeshGmshLoader name="meshLoader" filename="mesh/liver.msh" />
                <TetrahedronSetTopologyContainer name="topo" src="@meshLoader" />
                <MechanicalObject name="dofs" src="@meshLoader" />
                <TetrahedronSetGeometryAlgorithms template="Vec3" name="GeomAlgo" />
                <DiagonalMass  name="computed using mass density" massDensity="1" />
                <TetrahedralCorotationalFEMForceField template="Vec3" name="FEM" method="large" poissonRatio="0.3" youngModulus="3000" computeGlobalMatrix="0" />
                <FixedProjectiveConstraint  name="FixedProjectiveConstraint" indices="3 39 64" />
                <Node name="Visu" tags="Visual" gravity="0 -9.81 0">
                    <OglModel  name="VisualModel" src="@../../LiverSurface" />
                    <BarycentricMapping name="visual mapping" input="@../dofs" output="@VisualModel" />
                </Node>
                <Node name="Surf" gravity="0 -9.81 0">
                    <SphereLoader filename="mesh/liver.sph" />
                    <MechanicalObject name="spheres" position="@[-1].position" />
                    <SphereCollisionModel name="CollisionModel" listRadius="@[-2].listRadius"/>
                    <BarycentricMapping name="sphere mapping" input="@../dofs" output="@spheres" />
                </Node>
            </Node>
        </Node>
        )";
        
        executeInParallel(sceneStr.c_str(), nbScenes, nbSteps);
    }
    
    void testParallelCaduceusNoMT(const std::size_t nbScenes, const std::size_t nbSteps)
    {
        const std::string sceneStr = R"(
        <?xml version="1.0" ?>
        <Node name="root" gravity="0 -1000 0" dt="0.04">
            <Node name="RequiredPlugins">
                <RequiredPlugin name="Sofa.Component.AnimationLoop"/> <!-- Needed to use components [FreeMotionAnimationLoop] -->
                <RequiredPlugin name="Sofa.Component.Collision.Detection.Algorithm"/> <!-- Needed to use components [BVHNarrowPhase BruteForceBroadPhase CollisionPipeline] -->
                <RequiredPlugin name="Sofa.Component.Collision.Detection.Intersection"/> <!-- Needed to use components [MinProximityIntersection] -->
                <RequiredPlugin name="Sofa.Component.Collision.Geometry"/> <!-- Needed to use components [LineCollisionModel PointCollisionModel TriangleCollisionModel] -->
                <RequiredPlugin name="Sofa.Component.Collision.Response.Contact"/> <!-- Needed to use components [CollisionResponse] -->
                <RequiredPlugin name="Sofa.Component.Constraint.Lagrangian.Correction"/> <!-- Needed to use components [UncoupledConstraintCorrection] -->
                <RequiredPlugin name="Sofa.Component.Constraint.Lagrangian.Solver"/> <!-- Needed to use components [LCPConstraintSolver] -->
                <RequiredPlugin name="Sofa.Component.IO.Mesh"/> <!-- Needed to use components [MeshOBJLoader] -->
                <RequiredPlugin name="Sofa.Component.LinearSolver.Iterative"/> <!-- Needed to use components [CGLinearSolver] -->
                <RequiredPlugin name="Sofa.Component.Mapping.Linear"/> <!-- Needed to use components [BarycentricMapping] -->
                <RequiredPlugin name="Sofa.Component.Mass"/> <!-- Needed to use components [UniformMass] -->
                <RequiredPlugin name="Sofa.Component.ODESolver.Backward"/> <!-- Needed to use components [EulerImplicitSolver] -->
                <RequiredPlugin name="Sofa.Component.SolidMechanics.FEM.Elastic"/> <!-- Needed to use components [HexahedronFEMForceField] -->
                <RequiredPlugin name="Sofa.Component.StateContainer"/> <!-- Needed to use components [MechanicalObject] -->
                <RequiredPlugin name="Sofa.Component.Topology.Container.Constant"/> <!-- Needed to use components [MeshTopology] -->
                <RequiredPlugin name="Sofa.Component.Topology.Container.Grid"/> <!-- Needed to use components [SparseGridRamificationTopology] -->
                <RequiredPlugin name="Sofa.Component.Visual"/> <!-- Needed to use components [InteractiveCamera VisualStyle] -->
                <RequiredPlugin name="Sofa.GL.Component.Rendering3D"/> <!-- Needed to use components [OglModel] -->
                <RequiredPlugin name="Sofa.GL.Component.Shader"/> <!-- Needed to use components [LightManager SpotLight] -->
                <RequiredPlugin name="Sofa.Component.LinearSystem"/> <!-- Needed to use components [MatrixLinearSystem] -->
            </Node>
            
            <FreeMotionAnimationLoop parallelCollisionDetectionAndFreeMotion="false" />
            <VisualStyle displayFlags="showVisual  " /> <!--showBehaviorModels showCollisionModels-->
            <LCPConstraintSolver tolerance="1e-3" maxIt="1000" initial_guess="false" build_lcp="false"  printLog="0" mu="0.2"/>
            <CollisionPipeline depth="15" verbose="0" draw="0" />
            <BruteForceBroadPhase/>
            <BVHNarrowPhase/>
            <MinProximityIntersection name="Proximity" alarmDistance="1.5" contactDistance="1" />

            <CollisionResponse name="Response" response="FrictionContactConstraint" />

            <InteractiveCamera position="0 30 90" lookAt="0 30 0"/>

            <MeshOBJLoader name="visual_snake_body" filename="mesh/snake_body.obj" handleSeams="1" />
            <MeshOBJLoader name="visual_snake_cornea" filename="mesh/snake_cornea.obj" handleSeams="1" />
            <MeshOBJLoader name="visual_snake_eye" filename="mesh/snake_yellowEye.obj" handleSeams="1" />
            <MeshOBJLoader name="SOFA_pod" filename="mesh/SOFA_pod.obj" handleSeams="1" />

            <Node name="Snake" >

                <SparseGridRamificationTopology name="grid" n="4 12 3" fileTopology="mesh/snake_body.obj" nbVirtualFinerLevels="3" finestConnectivity="0"/>

                <EulerImplicitSolver name="cg_odesolver" rayleighMass="1" rayleighStiffness="0.03" />
                <MatrixLinearSystem template="CompressedRowSparseMatrixMat3x3" name="linearSystem"/>
                <CGLinearSolver name="linear_solver" iterations="20" tolerance="1e-12" threshold="1e-18" template="CompressedRowSparseMatrixMat3x3" linearSystem="@linearSystem"/>
                <MechanicalObject name="dofs"  scale="1" dy="2" position="@grid.position"  tags="NoPicking" />
                <UniformMass totalMass="1.0" />
                <HexahedronFEMForceField name="FEM" youngModulus="30000.0" poissonRatio="0.3" method="large" updateStiffnessMatrix="false" printLog="0" />
                <UncoupledConstraintCorrection defaultCompliance="184" useOdeSolverIntegrationFactors="0"/>

                <Node name="Collis">
                    <MeshOBJLoader name="loader" filename="mesh/meca_snake_900tri.obj" />
                    <MeshTopology src="@loader" />
                    <MechanicalObject src="@loader" name="CollisModel" />
                    <TriangleCollisionModel  selfCollision="0" />
                    <LineCollisionModel    selfCollision="0" />
                    <PointCollisionModel  selfCollision="0" />
                    <BarycentricMapping input="@.." output="@." />
                </Node>

                <Node name="VisuBody" tags="Visual" >
                    <OglModel  name="VisualBody" src="@../../visual_snake_body" texturename="textures/snakeColorMap.png"  />
                    <BarycentricMapping input="@.." output="@VisualBody" />
                </Node>

                <Node name="VisuCornea" tags="Visual" >
                    <OglModel  name="VisualCornea" src="@../../visual_snake_cornea"   />
                    <BarycentricMapping input="@.." output="@VisualCornea" />
                </Node>

                <Node name="VisuEye" tags="Visual" >
                    <OglModel  name="VisualEye" src="@../../visual_snake_eye"   />
                    <BarycentricMapping input="@.." output="@VisualEye" />
                </Node>
            </Node>

            <Node name="Base" >
                <Node name="Stick">
                         <MeshOBJLoader name="loader" filename="mesh/collision_batons.obj" />
                         <MeshTopology src="@loader" />
                         <MechanicalObject src="@loader" name="CollisModel" />
                         <LineCollisionModel simulated="false" moving="false" />
                         <PointCollisionModel simulated="false"  moving="false"/>
                </Node>
                <Node name="Blobs">
                         <MeshOBJLoader name="loader" filename="mesh/collision_boules_V3.obj" />
                         <MeshTopology src="@loader" />
                         <MechanicalObject src="@loader" name="CollisModel" />
                         <TriangleCollisionModel simulated="false" moving="false"/>
                         <LineCollisionModel simulated="false" moving="false"/>
                         <PointCollisionModel simulated="false" moving="false"/>
                </Node>

                    <Node name="Foot">
                    <MeshOBJLoader name="loader" filename="mesh/collision_pied.obj" />
                    <MeshTopology src="@loader" />
                    <MechanicalObject src="@loader" name="CollisModel" />
                    <TriangleCollisionModel simulated="false" moving="false"/>
                    <LineCollisionModel simulated="false" moving="false"/>
                    <PointCollisionModel simulated="false" moving="false"/>
                </Node>

                <Node name="Visu" tags="Visual" >
                    <OglModel  name="OglModel" src="@../../SOFA_pod"  />
                </Node>
            </Node>
        </Node>
        )";
        
        executeInParallel(sceneStr.c_str(), nbScenes, nbSteps);
    }
};

TEST_F(ParallelScenesTest , testParallelLiver )
{
    this->testParallelLiver(8, 5000);
}

TEST_F(ParallelScenesTest , testParallelCaduceusNoMT )
{
    this->testParallelCaduceusNoMT(8, 5000);
}
