#include <QApplication>
#include "QtMViewer.h"
#include <GL/glut.h>

// ---------------------------------------------------------------------
// Sofa interface
#include <sofa/helper/ArgumentParser.h>
#include <SofaSimpleGUI/SofaGlInterface.h>
sofa::newgui::SofaGlInterface sofaScene;     ///< The interface of the application with Sofa
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
  glutInit(&argc,argv);

  std::string fileName = "examples/oneTet.scn";
  sofa::helper::parse("Simple glut application featuring a Sofa scene.")
          .option(&sofaScene.plugins,'l',"load","load given plugins")
          .option(&fileName,'f',"file","scene file to load")
          (argc,argv);

  // --- Init sofa ---
  sofaScene.debug = false;
  sofaScene.init(fileName);

  // Read command lines arguments.
  QApplication application(argc,argv);

  // Instantiate the viewer.
  QtMViewer viewer(&sofaScene);

  viewer.setWindowTitle("simpleViewer");

  // Make the viewer window visible on screen.
  viewer.show();

  // Run main loop.
  return application.exec();
}
