
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/VecId.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/accessor.h>


using namespace sofa::simulation::tree;
using sofa::component::odesolver::EulerSolver;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;


using namespace sofa::simulation::tree;
using namespace sofa::component::odesolver;
// using namespace sofa::component::container;
// using namespace sofa::component::mass;
// using namespace sofa::component::visualmodel;
// using namespace sofa::component::collision;
// using namespace sofa::component::mapping;
// using namespace sofa::component::constraintset;
// using namespace sofa::component::projectiveconstraintset;
// using namespace sofa::component::controller;
// using namespace sofa::component::forcefield;
// using namespace sofa::component::fem::material;
// using namespace sofa::component::fem::forcefield;
// using namespace sofa::component::animationloop;
// using namespace sofa::component::linearsolver;
// using namespace sofa::component::topology;
// using namespace sofa::component::interactionforcefield;
// using namespace sofa::component::engine;
// using namespace sofa::component::behaviormodel::eulerianfluid;
// using namespace sofa::component::misc;
// using namespace sofa::component::configurationsetting;
// using namespace sofa::component::loader;
using namespace sofa::core::objectmodel;
using namespace sofa::core::visual;
using namespace sofa::component::visualmodel;
using namespace sofa::helper;
using namespace sofa::gui;
using namespace sofa::simulation;

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    parse("This is a SOFA application.")
    (argc,argv);
    initMain();
    GUIManager::Init(argv[0]);
//    GUIManager::SetFullScreen();

    // The graph root node
    setSimulation(new TreeSimulation());
    Node::SPtr groot = tree::getSimulation()->createNewGraph("root");
    groot->setGravity( Coord3(0,-10,0) );

    // One solver for all the graph
    EulerSolver::SPtr solver = New<EulerSolver>();
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // One node to define the particle
    Node::SPtr particule_node = groot.get()->createChild("particle_node");
    // The particule, i.e, its degrees of freedom : a point with a velocity
    MechanicalObject3::SPtr dof = New<MechanicalObject3>();
    dof->setName("particle");
    particule_node->addObject(dof);
    dof->resize(1);
    // get write access the particle positions vector
    WriteAccessor< Data<MechanicalObject3::VecCoord> > positions = *dof->write( VecId::position() );
    positions[0] = Coord3(0,0,0);
    // get write access the particle velocities vector
    WriteAccessor< Data<MechanicalObject3::VecDeriv> > velocities = *dof->write( VecId::velocity() );
    velocities[0] = Deriv3(0,0,0);
    // show the particle
    dof->showObject.setValue(true);
    dof->showObjectScale.setValue(10.);

    // Its properties, i.e, a simple mass node
    UniformMass3::SPtr mass = New<UniformMass3>();
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    // Display Flags
    VisualStyle::SPtr style =
        New<VisualStyle>();
    groot->addObject(style);
    DisplayFlags& flags = *style->displayFlags.beginEdit();
    flags.setShowBehaviorModels(true);
    style->displayFlags.endEdit();

    tree::getSimulation()->init(groot.get());
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
    GUIManager::createGUI(groot);
    // GUIManager::SetDimension(800,800);
    GUIManager::MainLoop();

    return 0;
}
