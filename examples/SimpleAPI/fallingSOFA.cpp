
#include <sofa/core/IntrusiveObject.h>
#include <sofa/Modules.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/common/init.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/gui/init.h>
#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/common/GUIManager.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/Utils.h>
#include <vector>
#include <string>


sofa::simulation::Node::SPtr createScene(const sofa::simpleapi::Simulation::SPtr simu)
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(simu, "root") ;

    root->setGravity( sofa::type::Vec3(0,0,-9.81) );
    root->setAnimate(false);
    root->setDt(0.005);

    sofa::simpleapi::importPlugin(Sofa.Component.AnimationLoop);
    sofa::simpleapi::importPlugin(Sofa.Component.AnimationLoop);
    sofa::simpleapi::importPlugin(Sofa.Component.Collision.Detection.Algorithm);
    sofa::simpleapi::importPlugin(Sofa.Component.Collision.Detection.Intersection);
    sofa::simpleapi::importPlugin(Sofa.Component.Collision.Geometry);
    sofa::simpleapi::importPlugin(Sofa.Component.Collision.Response.Contact);
    sofa::simpleapi::importPlugin(Sofa.Component.Constraint.Lagrangian.Correction);
    sofa::simpleapi::importPlugin(Sofa.Component.Constraint.Lagrangian.Solver);
    sofa::simpleapi::importPlugin(Sofa.Component.LinearSolver.Direct);
    sofa::simpleapi::importPlugin(Sofa.Component.LinearSolver.Iterative);
    sofa::simpleapi::importPlugin(Sofa.Component.Mapping.Linear);
    sofa::simpleapi::importPlugin(Sofa.Component.Mass);
    sofa::simpleapi::importPlugin(Sofa.Component.ODESolver.Backward);
    sofa::simpleapi::importPlugin(Sofa.Component.SolidMechanics.FEM.Elastic);
    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);
    sofa::simpleapi::importPlugin(Sofa.Component.Topology.Container.Dynamic);
    sofa::simpleapi::importPlugin(Sofa.Component.Topology.Container.Grid);
    sofa::simpleapi::importPlugin(Sofa.Component.Topology.Mapping);
    sofa::simpleapi::importPlugin(Sofa.Component.Visual);
    sofa::simpleapi::importPlugin(Sofa.GL.Component.Rendering3D);
    sofa::simpleapi::importPlugin(Sofa.GUI.Component);
    sofa::simpleapi::importPlugin(Sofa.Component.IO.Mesh);
    sofa::simpleapi::importPlugin("MultiThreading");

    sofa::simpleapi::createObject(root, "VisualStyle", {{"displayFlags", "showVisual"}});
    sofa::simpleapi::createObject(root, "ConstraintAttachButtonSetting");
    sofa::simpleapi::createObject(root, "FreeMotionAnimationLoop");
    sofa::simpleapi::createObject(root, "ProjectedGaussSeidelConstraintSolver",{{"maxIterations","50"}, {"tolerance","1.0e-6"}});

    sofa::simpleapi::createObject(root, "CollisionPipeline",{{"name","Pipeline"}});
    sofa::simpleapi::createObject(root, "ParallelBruteForceBroadPhase",{{"name","BroadPhase"}});
    sofa::simpleapi::createObject(root, "ParallelBVHNarrowPhase",{{"name","NarrowPhase"}});
    sofa::simpleapi::createObject(root, "CollisionResponse",{{"name","ContactManager"},
                                                              {"response","FrictionContactConstraint"},
                                                              {"responseParams","mu=0.3"}});
    sofa::simpleapi::createObject(root, "NewProximityIntersection",{{"name","Intersection"},
                                                                     {"alarmDistance","0.02"},
                                                                     {"contactDistance","0.002"}});

    //Simulated Topology creation node
    const sofa::simulation::Node::SPtr BeamDomainFromGridTopology = sofa::simpleapi::createChild(root,"BeamDomainFromGridTopology");
    sofa::simpleapi::createObject(BeamDomainFromGridTopology,"RegularGridTopology", {{ "name","HexaTop"}, {"n","15 3 6"}, {"min","0 0.02 0"}, {"max","0.5 0.08 0.22"}});
    sofa::simpleapi::createObject(BeamDomainFromGridTopology,"TetrahedronSetTopologyContainer", {{ "name","Container"}, {"position","@HexaTop.position"}});
    sofa::simpleapi::createObject(BeamDomainFromGridTopology,"TetrahedronSetTopologyModifier", {{ "name","Modifier"}});
    sofa::simpleapi::createObject(BeamDomainFromGridTopology,"Hexa2TetraTopologicalMapping", {{"input","@HexaTop"}, {"output","@Container"}, {"swapping","true"}});

    //Simulated node
    const sofa::simulation::Node::SPtr FEMechanicalModel = sofa::simpleapi::createChild(root,"FE-MechanicalModel");
    sofa::simpleapi::createObject(FEMechanicalModel, "EulerImplicitSolver");
    sofa::simpleapi::createObject(FEMechanicalModel, "SparseLDLSolver", {{"name","ldl"}, {"template","CompressedRowSparseMatrixMat3x3"}, {"parallelInverseProduct","true"}} );
    sofa::simpleapi::createObject(FEMechanicalModel, "TetrahedronSetTopologyContainer", {{"name","Container"},  {"position","@../BeamDomainFromGridTopology/HexaTop.position"}, {"tetrahedra","@../BeamDomainFromGridTopology/Container.tetrahedra"}});
    sofa::simpleapi::createObject(FEMechanicalModel, "TetrahedronSetTopologyModifier", {{"name","Modifier"}});
    sofa::simpleapi::createObject(FEMechanicalModel, "MechanicalObject", {{"name","mstate"}, {"template","Vec3d"}, {"src","@Container"}});
    sofa::simpleapi::createObject(FEMechanicalModel, "TetrahedronFEMForceField", {{"name","forceField"}, {"listening","true"}, {"youngModulus","2e4"}, {"poissonRatio","0.45"}, {"method","large"}});
    sofa::simpleapi::createObject(FEMechanicalModel, "MeshMatrixMass", {{"totalMass","1.2"}});

    const sofa::simulation::Node::SPtr FEMechanicalModel_Surface = sofa::simpleapi::createChild(FEMechanicalModel,"Surface");
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "TriangleSetTopologyContainer", {{"name","Container"}});
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "TriangleSetTopologyModifier", {{"name","Modifier"}});
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "Tetra2TriangleTopologicalMapping", {{"input","@../Container"}, {"output","@Container"}, {"flipNormals","false"}});
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "MechanicalObject", {{"name","dofs"},{"rest_position","@../mstate.rest_position"}});
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "TriangleCollisionModel", {{"name","Collision"},{"proximity","0.001"}, {"color","0.94117647058824 0.93725490196078 0.89411764705882"}} );
    sofa::simpleapi::createObject(FEMechanicalModel_Surface, "IdentityMapping", {{"name","SurfaceMapping"}});

    const std::vector<std::string > visuFiles{"mesh/SofaScene/LogoVisu.obj", "mesh/SofaScene/SVisu.obj", "mesh/SofaScene/O.obj", "mesh/SofaScene/FVisu.obj", "mesh/SofaScene/AVisu.obj"};
    const std::vector<std::string > visuColor{"0.7 .35 0 1.0", "0.7 0.7 0.7 1", "0.7 0.7 0.7 1", "0.7 0.7 0.7 1", "0.7 0.7 0.7 1"};
    const std::vector<std::string > nodeNames{"VisuLogo", "VisuS", "VisuO", "VisuF", "VisuA"};
    for(unsigned i=0; i<visuFiles.size(); ++i)
    {
        const sofa::simulation::Node::SPtr visuNode = sofa::simpleapi::createChild(FEMechanicalModel,nodeNames[i]);
        sofa::simpleapi::createObject(visuNode, "MeshOBJLoader", {{"name","SurfaceLoader"}, {"filename",visuFiles[i]}, {"scale3d","0.015 0.015 0.015"}, {"translation","0 0.05 0"}, {"rotation","180 0 0"}});
        sofa::simpleapi::createObject(visuNode, "OglModel", {{"name","VisualModel"}, {"color",visuColor[i]}, {"position","@SurfaceLoader.position"}, {"triangles","@SurfaceLoader.triangles"}} );
        sofa::simpleapi::createObject(visuNode, "BarycentricMapping", {{"name","MappingVisu"}, {"input","@../mstate"}, {"output","@VisualModel"}, {"isMechanical","false"}} );
    }

    sofa::simpleapi::createObject(FEMechanicalModel, "LinearSolverConstraintCorrection", {{"linearSolver","@ldl"}});

    const sofa::simulation::Node::SPtr Floor = sofa::simpleapi::createChild(root,"Floor", {{"tags","NoBBox"}});
    sofa::simpleapi::createObject(Floor, "VisualStyle", {{"displayFlags","showCollisionModels"}});
    sofa::simpleapi::createObject(Floor, "TriangleSetTopologyContainer", {{"name","FloorTopo"},
                                                                          {"position","0.2 0 -0.5  0.2 0.1 -0.5  0.3 0.1 -0.5  0.3 0 -0.5  0.2 0 -0.6  0.2 0.1 -0.6  0.3 0.1 -0.6  0.3 0 -0.6"},
                                                                          {"triangles","0 2 1  0 3 2  0 1 5  0 5 4  0 4 7  0 7 3  1 2 6  1 6 5  3 7 6  3 6 2  4 5 6  4 6 7"}} );
    sofa::simpleapi::createObject(Floor, "MechanicalObject", {{"template","Vec3"}});
    sofa::simpleapi::createObject(Floor, "TriangleCollisionModel", {{"name","FloorCM"}, {"proximity","0.001"}, {"moving","0"}, {"simulated","0"}} );
    return root;
}

int main(int /**argc**/, char** argv)
{
    sofa::simulation::common::init();
    sofa::simulation::graph::init();

    const sofa::simpleapi::Simulation::SPtr simu = sofa::simpleapi::createSimulation("DAG") ;
    const auto root = createScene(simu);

    sofa::simulation::node::initRoot(root.get());

    sofa::gui::common::BaseGUI::setConfigDirectoryPath(sofa::helper::Utils::getSofaPathPrefix() + "/config", true);
    sofa::gui::common::BaseGUI::setScreenshotDirectoryPath(sofa::helper::Utils::getSofaPathPrefix() + "/screenshots", true);

    sofa::helper::system::PluginManager::getInstance().loadPlugin("SofaImGui");
    sofa::helper::system::PluginManager::getInstance().init();
    sofa::gui::init();

    if (int err = sofa::gui::common::GUIManager::Init(argv[0],"imgui")) return err;
    if (int err=sofa::gui::common::GUIManager::createGUI(nullptr)) return err;
    sofa::gui::common::GUIManager::SetDimension(800,600);
    sofa::gui::common::GUIManager::CenterWindow();

    sofa::gui::common::GUIManager::SetScene(root);

    if (int err = sofa::gui::common::GUIManager::MainLoop(root))
        return err;

    if (root!=nullptr)
        sofa::simulation::node::unload(root);


    sofa::gui::common::GUIManager::closeGUI();
    sofa::simulation::common::cleanup();
    sofa::simulation::graph::cleanup();
    return 0;
}
