#include <iostream>
#include "Sofa/Components/Scene.h"
#include "Sofa/GUI/FLTK/Main.h"

using std::cout;
using std::cerr;
using std::endl;

//----------------------------------------
// Stuff used in procedural scene building
#include "Sofa/Components/MassObject.h"
#include "Sofa/Components/EulerSolver.h"

using namespace Sofa::Components;
using namespace Sofa::Core;
using namespace Sofa::GUI::FLTK;
typedef Sofa::Components::Common::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;

// Stuff used in procedural scene building
//----------------------------------------


Scene* buildScene()
{
    Scene* scene = new Scene;
    scene->setDt(0.04);

    MechanicalGroup* group = new MechanicalGroup;
    scene->addBehaviorModel(group);
    group->setSolver( new EulerSolver );

    MassObject<MyTypes>* particles = new MassObject<MyTypes>;
    group->addObject(particles);
    scene->addVisualModel(particles);                      // make the particles visible
    particles->setGravity( Vec3( 0,-1,0 ) );
    particles->addMass( Vec3(2,0,0), Vec3(0,0,0), 1 );
    particles->addMass( Vec3(3,0,0), Vec3(0,0,0), 1 );

    scene->init();
    return scene;
}

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    Scene * scene=0;

    if (argc >= 2)
    {
        scene = Sofa::Components::Scene::loadScene(argv[1]);
    }

    if( scene==NULL )
    {
        cerr<<"Could not read file, build scene procedurally"<<endl;
        scene = buildScene();
    }

    Sofa::GUI::FLTK::MainLoop(argv[0]);
    return 0;
}
