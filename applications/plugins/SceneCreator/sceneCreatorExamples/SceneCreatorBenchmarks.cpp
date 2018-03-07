/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SceneCreator/SceneCreator.h>
#include <sofa/helper/ArgumentParser.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>

#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>


void fallingCubeExample(sofa::simulation::Node::SPtr root)
{
    //Add objects
    for (unsigned int i=0; i<10; ++i)
        sofa::modeling::addCube(root, "cubeFEM_"+ std::to_string(i), sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                10, 1000, 0.45,
                                sofa::defaulttype::Vec3Types::Deriv(0, (i+1)*5, 0));

    // Add floor
    sofa::modeling::addRigidPlane(root, "Floor", sofa::defaulttype::Vec3Types::Deriv(50, 1, 50),
                                  sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(40, 0, 40));
}

void fallingCylinderExample(sofa::simulation::Node::SPtr root)
{
    //Add objects
    for (unsigned int i=0; i<10; ++i)
        sofa::modeling::addCylinder(root, "cylinderFEM_"+ std::to_string(i), sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                    sofa::defaulttype::Vec3Types::Deriv(0, 1, 0), 1.0, 3.0,
                                    10, 1000, 0.45,
                                    sofa::defaulttype::Vec3Types::Deriv(0, (i+1)*5, 0));

    // Add floor
    sofa::modeling::addRigidPlane(root, "Floor", sofa::defaulttype::Vec3Types::Deriv(50, 1, 50),
                                  sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(40, 0, 40));
}

void fallingSphereExample(sofa::simulation::Node::SPtr root)
{
    //Add objects
    for (unsigned int i=0; i<10; ++i)
        sofa::modeling::addSphere(root, "sphereFEM_"+ std::to_string(i), sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                  sofa::defaulttype::Vec3Types::Deriv(0, 1, 0), 1.0,
                                  10, 1000, 0.45,
                                  sofa::defaulttype::Vec3Types::Deriv(0, (i+1)*5, 0));

    // Add floor
    sofa::modeling::addRigidPlane(root, "Floor", sofa::defaulttype::Vec3Types::Deriv(50, 1, 50),
                                  sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(40, 0, 40));
}


void fallingDrapExample(sofa::simulation::Node::SPtr root)
{
    //Add objects
    for (unsigned int i=0; i<6; ++i){
        sofa::modeling::addRigidCube(root, "cubeFIX_"+ std::to_string(i), sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                     sofa::defaulttype::Vec3Types::Deriv(-8.0+i*4, 0.51, 0));

        sofa::modeling::addRigidCylinder(root, "cylinderFIX_"+ std::to_string(i), sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                         sofa::defaulttype::Vec3Types::Deriv(1, 0, 0), 0.5, 3.0,
                                         sofa::defaulttype::Vec3Types::Deriv(0, 0.51, -10.0+i*4));
    }

    // Add floor
    sofa::modeling::addRigidPlane(root, "Floor", sofa::defaulttype::Vec3Types::Deriv(50, 1, 50),
                                  sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(40, 0, 40));

    // Add falling plane
    sofa::modeling::addPlane(root, "Drap", sofa::defaulttype::Vec3Types::Deriv(50, 1, 50), 30, 600, 0.3,
                             sofa::defaulttype::Vec3Types::Deriv(0, 30, 0), sofa::defaulttype::Vec3Types::Deriv(0, 0, 0), sofa::defaulttype::Vec3Types::Deriv(20, 0, 20));
}


int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();

    bool showHelp = false;
    unsigned int idExample = 0;
    ArgumentParser* argParser = new ArgumentParser(argc, argv);
    argParser->addArgument(po::value<bool>(&showHelp)->default_value(false)->implicit_value(true),                  "help,h", "Display this help message");
    argParser->addArgument(po::value<unsigned int>(&idExample)->default_value(0)->notifier([](unsigned int value)
    {
        if (value < 0 || value > 9) {
            std::cerr << "Example Number to enter from (0 - 9), current value: " << value << std::endl;
            exit( EXIT_FAILURE );
        }
    }),                                                                                                             "example,e", "Example Number to enter from (0 - 9)");

    argParser->parse();

    if(showHelp)
    {
        argParser->showHelp();
        exit( EXIT_SUCCESS );
    }

    // init GUI
    sofa::gui::initMain();
    sofa::gui::GUIManager::Init(argv[0]);

    // Create simulation tree
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());


    // Create the graph root node with collision
    sofa::simulation::Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    root->setGravity( sofa::defaulttype::Vec3Types::Deriv(0,-10.0,0) );


    // Create scene example (depends on user input)
    switch (idExample)
    {
    case 0:
        fallingCubeExample(root);
        break;
    case 1:
        fallingCylinderExample(root);
        break;
    case 2:
        fallingSphereExample(root);
        break;
    case 3:
        fallingDrapExample(root);
        break;
    default:
        fallingCubeExample(root);
        break;
    }


    root->setAnimate(false);

    sofa::simulation::getSimulation()->init(root.get());

    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    sofa::simulation::tree::cleanup();

    return 0;
}
