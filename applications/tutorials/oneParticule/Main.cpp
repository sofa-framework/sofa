#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/gui/SofaGUI.h>

using namespace sofa::simulation::tree;
using sofa::component::odesolver::EulerSolver;


//SOFA_DOUBLE (if you want to use only double in sofa), or
//SOFA_FLOAT  (if you want to use only float in sofa)
//can be set in sofa-default.cfg, and you only need to include <sofa/component/typedef/Sofa_typedef.h>
#ifndef SOFA_DOUBLE
#ifndef SOFA_FLOAT

#define SOFA_FLOAT //only using float
#endif
#endif


#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);
    sofa::gui::SofaGUI::Init(argv[0]);

    // The graph root node
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );

    // One solver for all the graph
    EulerSolver* solver = new EulerSolver;
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // One node to define the particle
    GNode* particule_node = new GNode;
    particule_node->setName("particle_node");
    groot->addChild( particule_node );

    // The particule, i.e, its degrees of freedom : a point with a velocity
    MechanicalObject3* particle = new MechanicalObject3;
    particle->setName("particle");
    particule_node->addObject(particle);
    particle->resize(1);
    // The point
    (*particle->getX())[0] = Coord3(0,0,0);
    // The velocity
    (*particle->getV())[0] = Coord3(0,0,0);

    // Its properties, i.e, a simple mass node
    UniformMass3* mass = new UniformMass3;
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    getSimulation()->init(groot);
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
    sofa::gui::SofaGUI::MainLoop(groot);

    return 0;
}
