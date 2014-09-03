/**
  A simple qt application featuring a Sofa simulation, using the newgui API.

  @author Francois Faure, 2014
  */

#include <QApplication>
#include <QResource>
#include "QSofaMainWindow.h"
#include <GL/glut.h>
#include <sofa/helper/ArgumentParser.h>
#include <string>
#include <iostream>

int main(int argc, char** argv)
{
    glutInit(&argc,argv);

    // access to the compiled resource file (icons etc.)
    // On linux, this file can be generated using the command line: rcc -binary SofaSimpleGUI.qrc -o SofaSimpleGUI.rcc
    // See http://qt-project.org/doc/qt-4.8/resources.html
    // TODO: create a build rule to update it automatically when needed
    // Note that icons can also be chosen by QStyle
    std::string path_to_resource = std::string(QTSOFA_SRC_DIR) + "/SofaSimpleGUI.rcc";
//    std::cout<<"path to resource = " << path_to_resource << std::endl;
    QResource::registerResource(path_to_resource.c_str());

    std::string fileName = "C:/MyFiles/Perso/Dev/imr/dep/sofa/examples/Demos/caduceus.scn";
    std::vector<std::string> plugins;
    sofa::helper::parse("Simple glut application featuring a Sofa scene.")
            .option(&plugins,'l',"load","load given plugins")
            .option(&fileName,'f',"file","scene file to load")
            (argc,argv);


    // Read command lines arguments.
    QApplication application(argc,argv);

    // Instantiate the main window and make it visible.
    QSofaMainWindow mainWindow;
    mainWindow.sofaScene.loadPlugins( plugins );
    mainWindow.initSofa(fileName);
    mainWindow.setWindowTitle("qtSofa");
    mainWindow.resize(800,600);
    mainWindow.show();
    mainWindow.sofaScene.play();


    // Run main loop.
    return application.exec();
}
