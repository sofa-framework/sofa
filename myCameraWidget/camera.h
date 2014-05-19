#include <libQGLViewer-2.5.2/QGLViewer/qglviewer.h>
#include <QtOpengl>
class Viewer : public QGLViewer
{
protected :
  virtual void draw();
  virtual void init();
  virtual QString helpString() const;

private :
  virtual void drawSpiral();
};
