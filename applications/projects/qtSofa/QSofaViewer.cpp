/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "QSofaViewer.h"
#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QTimer>
#include <QMessageBox>
#include <QAction>
#include <sofa/helper/system/glut.h>
#include <iostream>
using std::cout;
using std::endl;

GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

//GLfloat cp[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat camera_target[] = { 22.0, 0.0, 0.0 };
GLfloat camera_angle = 55;
GLfloat znear = 15;
GLfloat zfar = 35;
GLfloat DegToRad = 3.1415927 / 180;


QSofaViewer::QSofaViewer(QSofaScene *sofaScene, QGLWidget* contextSharing, QWidget *parent) :
    QGLWidget(parent, contextSharing), _sofaScene(sofaScene), _sofaGL(0)
{
    _drag = NULL;
	connect(sofaScene, SIGNAL(opened()), this, SLOT(reset()));
    connect(sofaScene, SIGNAL(stepEnd()), this, SLOT(draw()));

    {
        QAction* toggleFullScreenAct = new QAction( tr("&FullScreen"), this );
        toggleFullScreenAct->setShortcut(QKeySequence(Qt::CTRL+Qt::Key_F));
        toggleFullScreenAct->setShortcutContext(Qt::ApplicationShortcut);
        toggleFullScreenAct->setToolTip(tr("Show full screen"));
        connect(toggleFullScreenAct, SIGNAL(triggered()), this, SLOT(toggleFullScreen()));
        this->addAction(toggleFullScreenAct);
    }

}

void QSofaViewer::toggleFullScreen()
{
    std::cerr<<"QSofaViewer::toggleFullScreen()" << std::endl;
    if( this->isFullScreen() ){
        this->showNormal();
    }
    else {
        this->showFullScreen();
    }
}


void QSofaViewer::initializeGL()
{
    glClearColor (0.0, 0.0, 0.0, 0.0);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glEnable(GL_DEPTH_TEST);


    _camera.setlookAt (
            0,0,25,
            camera_target[0],camera_target[1],camera_target[2],
            0.0, 1.0, 0.0 // up vector
            );


}

void QSofaViewer::paintGL()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity ();
    _camera.lookAt();
//    gluLookAt (
//            cp[0],cp[1],cp[2],
//            camera_target[0],camera_target[1],camera_target[2],
//            0.0, 1.0, 0.0 // up vector
//            );

    // we need to init sofaGL here in order to be able to call initTextures (needing an active opengl context) for each opened scene
    if(!_sofaGL)
    {
        _sofaGL = new SofaGL(_sofaScene);
        viewAll();
    }

    _sofaGL->draw();

    // display a box, for debug
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);

}

void QSofaViewer::reset()
{
    delete _sofaGL;
    _sofaGL = 0;
}

void QSofaViewer::draw()
{
    update();
}

void QSofaViewer::viewAll()
{
    if(!_sofaGL)
        return;

//    SReal cp[3], ct[3], zn, zf;
//    Camera::Vec3 eye = _camera.eye();
//    for( int i=0; i<3; i++ ) cp[i] = eye[i];
//    _sofaGL->viewAll(
//            &cp[0],&cp[1],&cp[2],
//            &ct[0],&ct[1],&ct[2],
//            camera_angle * DegToRad, &zn, &zf
//            );
//    cout << "QSofaViewer::viewAll, camera eye before = " << _camera.eye().transpose() << endl << ", linear = " << endl << _camera.getTransform().linear().inverse() << endl;
//    cout << " cp = " << cp[0] << " " << cp[1] << " " << cp[2] << ", ct = "<< ct[0] << " " << ct[1] << " " << ct[2]  << endl;
//     _camera.setlookAt(cp[0],cp[1],cp[2],
//            ct[0],ct[1],ct[2],
//            _camera.getTransform().linear()(1,0), _camera.getTransform().linear()(1,1), _camera.getTransform().linear()(1,2)); // use current y direction as up axis
//    cout << "QSofaViewer::viewAll, camera eye after = " << _camera.eye().transpose() << endl << ", linear = " << endl << _camera.getTransform().linear().inverse() << endl;
//    znear = zn;
//    zfar = zf;

    float xmin, xmax, ymin, ymax, zmin, zmax;
    _sofaGL->getSceneBBox(&xmin,&ymin,&zmin, &xmax,&ymax,&zmax);
    _camera.viewAll(xmin,ymin,zmin, xmax,ymax,zmax);

    update();
}

void QSofaViewer::resizeGL(int w, int h)
{
    glViewport (0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    _camera.perspective(camera_angle, (GLfloat) w/(GLfloat) h, 0.01, 100);
    glMatrixMode (GL_MODELVIEW);
}

void QSofaViewer::keyPressEvent ( QKeyEvent * /*event*/ )
{
//    if( event->key() == Qt::Key_Shift ) cout << "Shift ";
//    else if( event->key() == Qt::Key_Control ) cout << "Control ";
//    else if( event->key() == Qt::Key_Alt ) cout << "Alt ";
//    cout << event->text().toStdString();
}

void QSofaViewer::keyReleaseEvent ( QKeyEvent * /*event*/ )
{
//    cout << endl;
}

void QSofaViewer::mousePressEvent ( QMouseEvent * event )
{
    if(!_sofaGL)
		return;

    if( isShiftPressed() )
    {
        if( isControlPressed() ) // pick an existing interactor
        {
            _drag = _sofaGL->pickInteractor(_camera.eye()[0], _camera.eye()[1], _camera.eye()[2], event->x(), event->y());
        }
        else if( sofa::simplegui::PickedPoint glpicked = _sofaGL->pick(
                     _camera.eye()[0], _camera.eye()[1], _camera.eye()[2],
                     event->x(),event->y() )  ) // create new interactor
        {
//            cout << "Picked: " << glpicked <<  endl;
            _drag = _sofaGL->getInteractor(glpicked);
            if( _drag == NULL )
            {
                cout << "create new interactor" << endl;
                _drag = new sofa::simplegui::SpringInteractor(glpicked,10000);
                _sofaGL->attach(_drag);
            }
            else cout << "reuse interactor" << endl;
        }
        else {
            cout << "no particle glpicked" << endl;
        }
    }
    else if(isControlPressed()){
        _sofaGL->glPick(event->x(), event->y());
        update();
    }
    else {
        if( _camera.handleMouseButton(
            event->button()==Qt::LeftButton ? Camera::ButtonLeft : event->button()==Qt::MiddleButton ? Camera::ButtonMiddle : Camera::ButtonRight ,
            Camera::ButtonDown,
            event->x(), event->y()
                    ))
        {
            return;
        }

    }

}

void QSofaViewer::mouseMoveEvent ( QMouseEvent * event )
{
    if(!_sofaGL)
		return;

    if( _drag != NULL )
    {
        _sofaGL->move(_drag, event->x(), event->y());
    }
    else if( _camera.handleMouseMotion(event->x(), event->y()) )
    {
        update();
        return;
    }

}

void QSofaViewer::mouseReleaseEvent ( QMouseEvent * event )
{
    if(!_sofaGL)
		return;

    if( _drag != NULL )
    {
        if(QApplication::keyboardModifiers() & Qt::ShiftModifier )
        {
            _sofaGL->detach(_drag);
            delete _drag;
//                cout << "delete interactor " << endl;
        }
        _drag = NULL;
    }
    else
    {
        if( _camera.handleMouseButton(
            event->button()==Qt::LeftButton ? Camera::ButtonLeft : event->button()==Qt::MiddleButton ? Camera::ButtonMiddle : Camera::ButtonRight ,
            Camera::ButtonUp,
            event->x(), event->y()
                    ))
        {
            return;
        }

    }
}
