#include "Sofa/Components/XML/SceneNode.h"
#include "Sofa/Components/XML/GroupNode.h"
#include "Sofa/Components/XML/SolverNode.h"
#include "Sofa/Components/XML/DynamicNode.h"
#include "Sofa/Components/MassObject.h"
#include "Sofa/Components/EulerSolver.h"
#include "Sofa/GUI/FLTK/Main.h"

using namespace Sofa::Components;
using namespace Sofa::Components::XML;
using namespace Sofa::Core;
using namespace Sofa::GUI::FLTK;
typedef Sofa::Components::Common::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;

int main(int argc, char** argv)
{
    SceneNode* scene = new SceneNode("The scene","default");
    scene->setAttribute("dt","0.04");

    GroupNode* group = new GroupNode("The group","default",scene);

    SolverNode* solver = new SolverNode("The Solver","RungeKutta4", group);

    DynamicNode* particles = new DynamicNode("The particles","MassObject",group);
    particles->setAttribute("gravity","0 -1 0");
    particles->setAttribute("positions","2 0 0  3 0 0");
    particles->setAttribute("velocities","0 0 0  0 0 0");
    particles->setAttribute("mass","1 1");


    scene->init();
    MainLoop(argv[0]);
    return 0;
}
