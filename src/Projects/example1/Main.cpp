#include <iostream>
#include "Sofa/Components/Scene.h"
#include "Sofa/GUI/FLTK/Main.h"

using std::cout;
using std::endl;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    const char* filename;

    if (argc >= 2)
    {
        filename = argv[1];
    }
    else
    {
        filename = "Data/demo6.scn";
    }

    Sofa::Components::Scene::loadScene(filename);
    Sofa::GUI::FLTK::MainLoop(argv[0]);
    return 0;
}
