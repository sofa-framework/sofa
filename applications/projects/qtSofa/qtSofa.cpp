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

