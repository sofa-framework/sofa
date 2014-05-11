#include "QSofaViewer.h"
#include <QApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QTimer>
#include <QMessageBox>
#include <GL/glut.h>
#include <iostream>
using std::cout;
using std::endl;

GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat camera_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat znear = camera_position[2]-10;
GLfloat zfar = camera_position[2]+10;


QSofaViewer::QSofaViewer(newgui::QSofaScene *sofaScene, QWidget *parent) :
    QGLWidget(parent), sofaGL(sofaScene)
{
    drag = NULL;
    connect(sofaScene, SIGNAL(stepEnd()), this, SLOT(draw()));
    connect(sofaScene, SIGNAL(opened()), this, SLOT(draw()));
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
    gluLookAt ( camera_position[0],camera_position[1],camera_position[2], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    sofaGL.draw();

    // display a box, for debug
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);

}

void QSofaViewer::draw()
{
//    sofaGL.sofaScene->step(0.04);
    update();
}

void QSofaViewer::resizeGL(int w, int h)
{
    glViewport (0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (55.0, (GLfloat) w/(GLfloat) h, znear, zfar );
    glMatrixMode (GL_MODELVIEW);
}

void QSofaViewer::keyPressEvent ( QKeyEvent * event )
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
    if(QApplication::keyboardModifiers() & Qt::ShiftModifier )
    {
        sofa::newgui::PickedPoint glpicked = sofaGL.pick(camera_position[0],camera_position[1],camera_position[2], event->x(), event->y() );
        if( glpicked )
        {
//            cout << "Picked: " << glpicked << endl;
            if( picked_to_interactor.find(glpicked)!=picked_to_interactor.end() ) // there is already an interactor on this particle
            {
                drag = picked_to_interactor[glpicked];
//                cout << "Re-use available interactor " << endl;
            }
            else {                                             // new interactor
                drag = picked_to_interactor[glpicked] = new sofa::newgui::SpringInteractor(glpicked,10000);
                sofaGL.attach(drag);
//                cout << "Create new interactor" << endl;
            }
//                sofaScene.printScene();
        }
        else {
            cout << "no particle glpicked" << endl;
        }
    }

}

void QSofaViewer::mouseMoveEvent ( QMouseEvent * event )
{
    if( drag != NULL )
    {
        sofaGL.move(drag, event->x(), event->y());
    }
}

void QSofaViewer::mouseReleaseEvent ( QMouseEvent * /*event*/ )
{
    if( drag != NULL )
    {
        if(QApplication::keyboardModifiers() & Qt::ShiftModifier )
        {
            sofaGL.detach(drag);
            delete drag;

            // remove it from the map
            Picked_to_Interactor::iterator i=picked_to_interactor.begin();
            while( i!=picked_to_interactor.end() && (*i).second != drag )
                i++;
            if( i!=picked_to_interactor.end() ){
//                cout << "Deleted interactor at " << (*i).first << endl;
                picked_to_interactor.erase(i);
//                cout << "new count of interactors: " << picked_to_interactor.size() << endl;
            }
            else assert( false && "Active interactor not found in the map" );

        }
        drag = NULL;
    }

}

