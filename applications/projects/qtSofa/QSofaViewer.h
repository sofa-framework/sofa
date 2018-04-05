/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef QTMVIEWER_H
#define QTMVIEWER_H
#include <GL/glew.h>
#include <QApplication>
#include <QGLWidget>
#include <SofaSimpleGUI/SofaScene.h>
#include <SofaSimpleGUI/SofaGL.h>
#include <SofaSimpleGUI/Camera.h>
#include "QSofaScene.h"

using sofa::simplegui::SofaScene;
using sofa::simplegui::SofaGL;
using sofa::simplegui::Interactor;
using sofa::simplegui::SpringInteractor;
using sofa::simplegui::PickedPoint;
using sofa::simplegui::Camera;

/**
 * @brief The QSofaViewer class is a Qt OpenGL viewer with a SofaGL interface to display and interact with a Sofa simulation.
 *
 * @author Francois Faure, 2014
 */
class QSofaViewer : public QGLWidget
{
    Q_OBJECT

public:
    explicit QSofaViewer(QSofaScene* sofaScene, QGLWidget* contextSharing, QWidget *parent = 0);
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
	// reset the display
	void reset();
    /// update the display
    void draw();
    /// adjust camera center and orientation to see the entire scene
    void viewAll();
    void toggleFullScreen();

protected:
	SofaScene* _sofaScene;	///< the sofa scene we want to display
    SofaGL* _sofaGL;		///< interface with the scene to display and pick in.

    Interactor* _drag; ///< current active interactor, NULL if none

    Camera _camera;  ///< viewpoint

    /// @return true iff SHIFT key is currently pressed
    inline bool isShiftPressed() const { return QApplication::keyboardModifiers() & Qt::ShiftModifier; }
    /// @return true iff CONTROL key is currently pressed
    inline bool isControlPressed() const { return QApplication::keyboardModifiers() & Qt::ControlModifier; }

};

#endif // QTVIEWER_H
