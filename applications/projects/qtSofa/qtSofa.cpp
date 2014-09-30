/**
  A simple qt application featuring a Sofa simulation, using the SofaSimpleGUI API.

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

    std::string fileName;
    // for some reason, setting a file at this point does not work. Using the default scene buit in C++ does.
//    fileName = std::string(QTSOFA_SRC_DIR) + "/../../../examples/Demos/caduceus.scn";
    std::vector<std::string> plugins;
    sofa::helper::parse("Simple qt application featuring a Sofa scene.")
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
