#include "Sofa/Components/Scene.h"
#include "Sofa/GUI/FLTK/Main.h"
//#include "Sofa/GUI/QT/Main.h"

using namespace Sofa::Components;
using namespace Sofa::GUI::FLTK;
//using namespace Sofa::GUI::QT;

int main(int argc, char** argv)
{
    Scene::loadScene((argc>=2)?argv[1]:"test1.scn");
    MainLoop(argv[0]);
    return 0;
}
