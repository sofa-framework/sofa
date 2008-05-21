#include "generateDoc.h"
#include <sofa/simulation/common/init.h>
#include <iostream>
#include <fstream>

int main(int /*argc*/, char** /*argv*/)
{
    sofa::simulation::init();
    std::cout << "Generating sofa-classes.html" << std::endl;
    projects::generateFactoryHTMLDoc("sofa-classes.html");
    std::cout << "Generating _classes.php" << std::endl;
    projects::generateFactoryPHPDoc("_classes.php","classes");
    return 0;
}
