/**
  A simple qt application featuring a Sofa simulation, using the newgui API.

  @author Francois Faure, 2014
  */

#include <QApplication>
#include "QSofaMainWindow.h"
#include <GL/glut.h>
#include <sofa/helper/ArgumentParser.h>

int main(int argc, char** argv)
{
  glutInit(&argc,argv);

  std::string fileName = "examples/oneTet.scn";
  std::vector<std::string> plugins;
  sofa::helper::parse("Simple glut application featuring a Sofa scene.")
          .option(&plugins,'l',"load","load given plugins")
          .option(&fileName,'f',"file","scene file to load")
          (argc,argv);


  // Read command lines arguments.
  QApplication application(argc,argv);

  // Instantiate the viewer.
  QSofaMainWindow mainWindow;
  mainWindow.initSofa(plugins,fileName);

  mainWindow.setWindowTitle("qtSofa");

  // Make the viewer window visible on screen.
  mainWindow.resize(800,600);
  mainWindow.show();
  mainWindow.start();

  // Run main loop.
  return application.exec();
}
