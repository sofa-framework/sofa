/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "generateDoc.h"
#include <SofaComponentMain/init.h>
#include <SofaSimulationTree/init.h>
#include <iostream>
#include <fstream>

int main(int /*argc*/, char** /*argv*/)
{
    sofa::simulation::tree::init();
    sofa::component::init();
    std::cout << "Generating sofa-classes.html" << std::endl;
    projects::generateFactoryHTMLDoc("sofa-classes.html");
    std::cout << "Generating _classes.php" << std::endl;
    projects::generateFactoryPHPDoc("_classes.php","classes");
    sofa::simulation::tree::cleanup();
    return 0;
}
