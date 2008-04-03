#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/core/objectmodel/Context.h>

#include <sofa/component/typedef/Mass_double.h>
#include <sofa/component/typedef/MechanicalObject_double.h>

#include <iostream>
#include <fstream>

#include <sofa/gui/SofaGUI.h>
//typedef Sofa::Components::Common::Vec3Types MyTypes;
typedef sofa::defaulttype::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);
    sofa::gui::SofaGUI::Init(argv[0]);

    // The graph root node
    sofa::simulation::tree::GNode* groot = new sofa::simulation::tree::GNode;
    groot->setName( "root" );

    // One solver for all the graph
    sofa::component::odesolver::EulerSolver* solver = new sofa::component::odesolver::EulerSolver;
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // Set gravity for all the graph
    sofa::component::contextobject::Gravity* gravity =  new sofa::component::contextobject::Gravity;
    gravity->setName("gravity");
    gravity->f_gravity.setValue( Vec3(0,-10,0) );
    groot->addObject(gravity);

    // One node to define the particle
    sofa::simulation::tree::GNode* particule_node = new sofa::simulation::tree::GNode;
    particule_node->setName("particle_node");
    groot->addChild( particule_node );

    // The particule, i.e, its degrees of freedom : a point with a velocity
    MechanicalObject3d* particle = new MechanicalObject3d;
    particle->setName("particle");
    particule_node->addObject(particle);
    particle->resize(1);
    // The point
    (*particle->getX())[0] = Vec3(0,0,0);
    // The velocity
    (*particle->getV())[0] = Vec3(0,0,0);

    // Its properties, i.e, a simple mass node
    UniformMass3d* mass = new UniformMass3d;
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    sofa::simulation::tree::getSimulation()->init(groot);
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
    sofa::gui::SofaGUI::MainLoop(groot);

    return 0;
}
