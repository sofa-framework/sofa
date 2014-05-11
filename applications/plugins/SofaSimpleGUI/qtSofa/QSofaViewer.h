#ifndef QTMVIEWER_H
#define QTMVIEWER_H
#include <GL/glew.h>
#include <QGLWidget>
#include <SofaSimpleGUI/SofaGLScene.h>
#include "QSofaScene.h"
using sofa::newgui::SofaGL;
using sofa::newgui::Interactor;
using sofa::newgui::SpringInteractor;
using sofa::newgui::PickedPoint;
#include <map>
using std::map;

/**
 * @brief The QSofaViewer class is a Qt OpenGL viewer with a SofaGL interface to display and interact with a Sofa simulation.
 *
 * @author Francois Faure, 2014
 */
class QSofaViewer : public QGLWidget
{
    Q_OBJECT

public:
    explicit QSofaViewer(sofa::newgui::QSofaScene* sofaScene, QWidget *parent = 0);
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
    /// update the display
    void draw();

protected:
    SofaGL sofaGL; ///< interface with the scene to display and pick in.

    typedef map< PickedPoint, Interactor*> Picked_to_Interactor;
    /** Currently available interactors, associated with picked points.
     *  The interactors are not necessarily being manipulated. Only one can be manipulated at at time.
     */
    Picked_to_Interactor picked_to_interactor;
    Interactor* drag;                            ///< The currently active interactor


};

#endif // QTVIEWER_H
