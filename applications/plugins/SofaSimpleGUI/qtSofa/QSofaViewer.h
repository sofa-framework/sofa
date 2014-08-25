#ifndef QTMVIEWER_H
#define QTMVIEWER_H
#include <GL/glew.h>
#include <QApplication>
#include <QGLWidget>
#include <SofaSimpleGUI/SofaGL.h>
#include "QSofaScene.h"
using sofa::simplegui::SofaGL;
using sofa::simplegui::Interactor;
using sofa::simplegui::SpringInteractor;
using sofa::simplegui::PickedPoint;

/**
 * @brief The QSofaViewer class is a Qt OpenGL viewer with a SofaGL interface to display and interact with a Sofa simulation.
 *
 * @author Francois Faure, 2014
 */
class QSofaViewer : public QGLWidget
{
    Q_OBJECT

public:
    explicit QSofaViewer(sofa::simplegui::QSofaScene* sofaScene, QWidget *parent = 0);
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
    /// adjust camera center and orientation to see the entire scene
    void viewAll();
    void toggleFullScreen();

protected:
    SofaGL _sofaGL; ///< interface with the scene to display and pick in.

    Interactor* _drag; ///< current active interactor, NULL if none

    /// @return true iff SHIFT key is currently pressed
    inline bool isShiftPressed() const { return QApplication::keyboardModifiers() & Qt::ShiftModifier; }
    /// @return true iff CONTROL key is currently pressed
    inline bool isControlPressed() const { return QApplication::keyboardModifiers() & Qt::ControlModifier; }

};

#endif // QTVIEWER_H
