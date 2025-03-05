/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include "../lib/SofaModeler.h"

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>

#include <SofaSimulationGraph/DAGSimulation.h>

#include <sofa/helper/logging/Messaging.h>

#include <SofaComponentAll/initSofaComponentAll.h>

#include <QApplication>

#include <iostream>
#include <fstream>

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    sofa::component::initSofaComponentAll();

    // TODO: create additional handlers depending on command-line parameters

    QApplication* application = new QApplication(argc, argv);
    (void)application;

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

	Q_INIT_RESOURCE(icons);
    sofa::gui::qt::SofaModeler* sofaModeler = new sofa::gui::qt::SofaModeler();

    //application->setMainWidget(sofaModeler);
    sofaModeler->show();

    std::string binaryName=argv[0];
#ifdef WIN32
    const std::string exe=".exe";
    if (binaryName.size() > exe.size()) binaryName = binaryName.substr(0, binaryName.size()-exe.size());
#endif
    if (!binaryName.empty() && binaryName[binaryName.size()-1] == 'd') sofaModeler->setDebugBinary(true);

    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/MODELER.png" )).c_str();
    application->setWindowIcon(QIcon(pathIcon));

    for (int i=1; i<argc; ++i)
    {
        //Try to open the simulations passed in command line
        sofaModeler->fileOpen(std::string(argv[i]));
    }
    if (argc <= 1 ) sofaModeler->newTab();

    int appReturnCode = application->exec();
    return appReturnCode;
}
