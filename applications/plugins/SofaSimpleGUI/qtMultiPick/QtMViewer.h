#ifndef QTMVIEWER_H
#define QTMVIEWER_H
#include <GL/glew.h>
#include <QGLWidget>
#include <SofaSimpleGUI/SofaGlInterface.h>
using sofa::newgui::SofaGlInterface;
using sofa::newgui::Interactor;
using sofa::newgui::SpringInteractor;
using sofa::newgui::PickedPoint;
#include <map>
using std::map;

class QtMViewer : public QGLWidget
{
    Q_OBJECT

public:
    explicit QtMViewer(sofa::newgui::SofaGlInterface* sofaScene, QGLWidget *parent = 0);
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);
    void keyPressEvent ( QKeyEvent * event );
    void keyReleaseEvent ( QKeyEvent * event );
    void mousePressEvent ( QMouseEvent * event );
    void mouseMoveEvent ( QMouseEvent * event );
    void mouseReleaseEvent ( QMouseEvent * event );


signals:

public slots:
    /// Apply one simulation time step
    void animate();

protected:
    // Sofa interface
    SofaGlInterface *sofaScene; ///< Sofa scenegraph and functions.

    typedef map< PickedPoint, Interactor*> Picked_to_Interactor;
    /** Currently available interactors, associated with picked points.
     *  The interactors are not necessarily being manipulated. Only one can be manipulated at at time.
     */
    Picked_to_Interactor picked_to_interactor;
    Interactor* drag;                            ///< The currently active interactor


};

#endif // QTVIEWER_H
