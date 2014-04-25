//#include "MainWindow.h"
//#include <QApplication>

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

//    return a.exec();
//}


#include <QApplication>
#include "QtViewer.h"
#include <GL/glut.h>

int main(int argc, char** argv)
{
  glutInit(&argc,argv);

  // Read command lines arguments.
  QApplication application(argc,argv);

  // Instantiate the viewer.
  QtViewer viewer;

  viewer.setWindowTitle("simpleViewer");

  // Make the viewer window visible on screen.
  viewer.show();

  // Run main loop.
  return application.exec();
}
