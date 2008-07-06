#include "generateDoc.h"
#include <sofa/component/init.h>
#include <iostream>
#include <fstream>

int main(int /*argc*/, char** /*argv*/)
{
    sofa::component::init();
    std::cout << "Generating sofa-classes.html" << std::endl;
    projects::generateFactoryHTMLDoc("sofa-classes.html");
    std::cout << "Generating _classes.php" << std::endl;
    projects::generateFactoryPHPDoc("_classes.php","classes");
    return 0;
}
