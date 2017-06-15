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
/**
  A simple qt application featuring a Sofa simulation, using the SofaSimpleGUI API.

  @author Francois Faure, 2014
  */

#include <QFile>
#include <QApplication>
#include <QResource>
#include "QSofaMainWindow.h"
#include <sofa/helper/system/glut.h>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <string>
#include <iostream>
#include <fstream>

#ifndef _DEBUG
const std::string CONFIG_FILE = "required_plugins_release.ini";
#else
const std::string CONFIG_FILE = "required_plugins_debug.ini";
#endif
/**
* @brief main application launch
*/
int main(int argc, char** argv)
{
    glutInit(&argc,argv);

	// Data
	std::string fileName;
	std::vector<std::string> plugins;

	// Read the config file to load the right plugins
	std::string path(CONFIG_FILE);
	path = std::string(QTSOFA_SRC_DIR) + "/config/" + path;
	path = sofa::helper::system::DataRepository.getFile(path);

	// Get the file content
	std::ifstream instream(path.c_str());
	std::string pluginPath;
	while(std::getline(instream,pluginPath))
		plugins.push_back(pluginPath);
	instream.close();

    // Load default sofa scene
    fileName = std::string(QTSOFA_SRC_DIR) + "/../../../examples/Demos/liver.scn";
	fileName = sofa::helper::system::DataRepository.getFile(fileName);
	
	// parse input data
    sofa::helper::parse("Simple qt application featuring a Sofa scene.")
            .option(&plugins,'l',"load","load given plugins")
            .option(&fileName,'f',"file","scene file to load")
            (argc,argv);

    // Read command lines arguments.
    QApplication application(argc,argv);

    // Instantiate the main window and make it visible.
    QSofaMainWindow mainWindow;
    mainWindow.sofaScene.loadPlugins( plugins );

	// Load plugin
	std::cout << "Plugin list : " << std::endl;
	for(unsigned int i=0; i<plugins.size(); i++) std::cout << " - " << plugins[i] << std::endl;

	// Init and show application
    mainWindow.initSofa(fileName);
    mainWindow.setWindowTitle("qtSofa");
    mainWindow.resize(800,600);
    mainWindow.show();
    mainWindow.sofaScene.play();


    // Run main loop.
    return application.exec();
}
