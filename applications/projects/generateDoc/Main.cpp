#include "generateDoc.h"
#include <sofa/simulation/tree/init.h>
#include <iostream>
#include <fstream>
#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#elif  SOFA_GUI_QTVIEWER
#include <sofa/gui/viewer/Main.h>
#elif  SOFA_GUI_QGLVIEWER
#include <sofa/gui/viewer/Main.h>
#elif  SOFA_GUI_QTOGREVIEWER
#include <sofa/gui/viewer/Main.h>
#endif

int main(int /*argc*/, char** /*argv*/)
{
    sofa::simulation::tree::init();
    std::cout << "Generating sofa-classes.html" << std::endl;
    projects::generateFactoryHTMLDoc("sofa-classes.html");
    std::cout << "Generating _classes.php" << std::endl;
    projects::generateFactoryPHPDoc("_classes.php","classes");
    return 0;
}
