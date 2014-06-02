#include "QSofaViewer.h"
#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QTimer>
#include <QMessageBox>
#include <QAction>
#include <GL/glut.h>
#include <iostream>
using std::cout;
using std::endl;

GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat cp[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat camera_target[] = { 2.0, 0.0, 0.0 };
GLfloat camera_angle = 55;
GLfloat znear = cp[2]-10;
GLfloat zfar = cp[2]+10;


QSofaViewer::QSofaViewer(newgui::QSofaScene *sofaScene, QWidget *parent) :
    QGLWidget(parent), _sofaGL(sofaScene)
{
    _drag = NULL;
    connect(sofaScene, SIGNAL(stepEnd()), this, SLOT(draw()));
    connect(sofaScene, SIGNAL(opened()), this, SLOT(draw()));

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
    cerr<<"QSofaViewer::toggleFullScreen()" << endl;
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

}

void QSofaViewer::paintGL()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity ();
    gluLookAt (
            cp[0],cp[1],cp[2],
            camera_target[0],camera_target[1],camera_target[2],
            0.0, 1.0, 0.0 // up vector
            );

    _sofaGL.draw();

    // display a box, for debug
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);

}

void QSofaViewer::draw()
{
    update();
}

void QSofaViewer::viewAll()
{
    SReal cp[3], ct[3], zn, zf;
    _sofaGL.viewAll(
            &cp[0],&cp[1],&cp[2],
            &ct[0],&ct[1],&ct[2],
            camera_angle, &zn, &zf
            );
    for( int i=0; i<3; i++ )
    {
        cp[i] = cp[i];
        camera_target[i] = ct[i];
    }
    znear = zn;
    zfar = zf;
}

void QSofaViewer::resizeGL(int w, int h)
{
    glViewport (0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (camera_angle, (GLfloat) w/(GLfloat) h, znear, zfar );
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
    if( isShiftPressed() )
    {
        if( isControlPressed() ) // pick an existing interactor
        {
            _drag = _sofaGL.pickInteractor(cp[0],cp[1],cp[2], event->x(), event->y());
        }
        else if( sofa::newgui::PickedPoint glpicked = _sofaGL.pick(cp[0],cp[1],cp[2], event->x(),event->y() )  ) // create new interactor
        {
//            cout << "Picked: " << glpicked <<  endl;
            _drag = _sofaGL.getInteractor(glpicked);
            if( _drag == NULL )
            {
                cout << "create new interactor" << endl;
                _drag = new sofa::newgui::SpringInteractor(glpicked,10000);
                _sofaGL.attach(_drag);
            }
            else cout << "reuse interactor" << endl;
        }
        else {
            cout << "no particle glpicked" << endl;
        }
    }

}

void QSofaViewer::mouseMoveEvent ( QMouseEvent * event )
{
    if( _drag != NULL )
    {
        _sofaGL.move(_drag, event->x(), event->y());
    }
}

void QSofaViewer::mouseReleaseEvent ( QMouseEvent * /*event*/ )
{
        if( _drag != NULL )
        {
            if(QApplication::keyboardModifiers() & Qt::ShiftModifier )
            {
                _sofaGL.detach(_drag);
                delete _drag;
//                cout << "delete interactor " << endl;
            }
            _drag = NULL;
        }
}

