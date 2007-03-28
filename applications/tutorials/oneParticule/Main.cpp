#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/objectmodel/Context.h>

#include <iostream>
#include <fstream>

#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#endif
#ifdef SOFA_GUI_QT
#include <sofa/gui/qt/Main.h>
#endif

//typedef Sofa::Components::Common::Vec3Types MyTypes;
typedef sofa::defaulttype::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    parse("This is a SOFA application.")
    (argc,argv);

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
    sofa::component::MechanicalObject<MyTypes>* particle = new sofa::component::MechanicalObject<MyTypes>;
    particle->setName("particle");
    particule_node->addObject(particle);
    particle->resize(1);
    // The point
    (*particle->getX())[0] = Vec3(0,0,0);
    // The velocity
    (*particle->getV())[0] = Vec3(0,0,0);

    // Its properties, i.e, a simple mass node
    sofa::component::mass::UniformMass<MyTypes,double>* mass = new sofa::component::mass::UniformMass<MyTypes,double>;
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    sofa::simulation::tree::Simulation::init(groot);
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
#ifdef SOFA_GUI_FLTK
    sofa::gui::fltk::MainLoop(argv[0],groot);
#endif
#ifdef SOFA_GUI_QT
    std::string fileName = "";
    sofa::gui::qt::MainLoop(argv[0],groot,fileName.c_str());
#endif
    return 0;
}
