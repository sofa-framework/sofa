/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include "viewer/qt/QtViewer.h"
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

#include <qevent.h>

//#ifdef __APPLE__
//#include <OpenGL.h>
//#endif

#include "GenGraphForm.h"


#include <sofa/helper/system/glut.h>
#include <sofa/helper/gl/glfont.h>
#include <sofa/helper/gl/RAII.h>
#ifdef SOFA_HAVE_GLEW
#include <sofa/helper/gl/GLSLShader.h>
#endif
#include <sofa/helper/io/ImageBMP.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/ColourPickingVisitor.h>

// define this if you want video and OBJ capture to be only done once per N iteration
//#define CAPTURE_PERIOD 5

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

namespace qt
{



using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;

using sofa::simulation::getSimulation;

helper::SofaViewerCreator< QtViewer> QtViewer_class("qt",false);
SOFA_DECL_CLASS ( QTGUI )

//Q:Why would the QtViewer write its .view file with the qglviewer (a GPL library) extension?
//A:The new QtViewer has the same parameters as QGLViewer.
//  Consequently, the old .view file is now totally incorrect.

///TODO: standardize .view file parameters
//const std::string QtViewer::VIEW_FILE_EXTENSION = "qglviewer.view";
const std::string QtViewer::VIEW_FILE_EXTENSION = "view";
// Mouse Interactor
bool QtViewer::_mouseTrans = false;
bool QtViewer::_mouseRotate = false;
Quaternion QtViewer::_mouseInteractorNewQuat;
Quaternion QtViewer::_newQuat;
Quaternion QtViewer::_currentQuat;
Vector3 QtViewer::_mouseInteractorRelativePosition(0, 0, 0);

//float g_DepthOffset[2] = { 3.0f, 0.0f };
float g_DepthOffset[2] =
{ 10.0f, 0.0f };
float g_DepthBias[2] =
{ 0.0f, 0.0f };

// These are the light's matrices that need to be stored
float g_mProjection[16] =
{ 0 };
float g_mModelView[16] =
{ 0 };
//float g_mCameraInverse[16] = {0};


QGLFormat QtViewer::setupGLFormat()
{
    QGLFormat f = QGLFormat::defaultFormat();
#if defined(QT_VERSION) && QT_VERSION >= 0x040200
    std::cout << "QtViewer: disabling vertical refresh sync" << std::endl;
    f.setSwapInterval(0); // disable vertical refresh sync
#endif
    return f;
}

// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
QtViewer::QtViewer(QWidget* parent, const char* name)
    : QGLWidget(setupGLFormat(), parent, name)
{

    groot = NULL;
    initTexturesDone = false;
#ifdef TRACKING_MOUSE
    m_grabActived = false;
#endif
    backgroundColour[0] = 1.0f;
    backgroundColour[1] = 1.0f;
    backgroundColour[2] = 1.0f;

    // setup OpenGL mode for the window
    //Fl_Gl_Window::mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA);
    timerAnimate = new QTimer(this);
    //connect( timerAnimate, SIGNAL(timeout()), this, SLOT(animate()) );

    _video = false;
    _axis = false;
    _background = 0;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;
    _waitForRender = false;
    sceneBBoxIsValid = false;

    /*_surfaceModel = NULL;
     _springMassView = NULL;
     _mapView = NULL;
     sphViewer = NULL;
     */

    // init trackball rotation matrix / quaternion
    _newTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);
    _newQuat = _newTrackball.GetQuaternion();

    ////////////////
    // Interactor //
    ////////////////
    m_isControlPressed = false;
    _mouseInteractorMoving = false;
    _mouseInteractorTranslationMode = false;
    _mouseInteractorRotationMode = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
#ifdef TRACKING
    savedX = 0;
    savedY = 0;
    firstTime = true;
    tracking = false;
#endif // TRACKING
    _mouseInteractorTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);
    _mouseInteractorNewQuat = _mouseInteractorTrackball.GetQuaternion();

    connect( &captureTimer, SIGNAL(timeout()), this, SLOT(captureEvent()) );
}

// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
QtViewer::~QtViewer()
{
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void QtViewer::initializeGL(void)
{
    static GLfloat specref[4];
    static GLfloat ambientLight[4];
    static GLfloat diffuseLight[4];
    static GLfloat specular[4];
    static GLfloat lmodel_ambient[] =
    { 0.0f, 0.0f, 0.0f, 0.0f };
    static GLfloat lmodel_twoside[] =
    { GL_FALSE };
    static GLfloat lmodel_local[] =
    { GL_FALSE };
    bool initialized = false;

    if (!initialized)
    {
        //std::cout << "progname=" << sofa::gui::qt::progname << std::endl;
        //sofa::helper::system::SetDirectory cwd(sofa::helper::system::SetDirectory::GetProcessFullPath(sofa::gui::qt::progname));

//#ifdef __APPLE__
//        std::cout << "QtViewer: disabling vertical refresh sync (Mac version)" << std::endl;
//        const GLint swapInterval = 0;
//        CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &swapInterval);
//#endif

        // Define light parameters
        //_lightPosition[0] = 0.0f;
        //_lightPosition[1] = 10.0f;
        //_lightPosition[2] = 0.0f;
        //_lightPosition[3] = 1.0f;

        _lightPosition[0] = -0.7f;
        _lightPosition[1] = 0.3f;
        _lightPosition[2] = 0.0f;
        _lightPosition[3] = 1.0f;

        ambientLight[0] = 0.5f;
        ambientLight[1] = 0.5f;
        ambientLight[2] = 0.5f;
        ambientLight[3] = 1.0f;

        diffuseLight[0] = 0.9f;
        diffuseLight[1] = 0.9f;
        diffuseLight[2] = 0.9f;
        diffuseLight[3] = 1.0f;

        specular[0] = 1.0f;
        specular[1] = 1.0f;
        specular[2] = 1.0f;
        specular[3] = 1.0f;

        specref[0] = 1.0f;
        specref[1] = 1.0f;
        specref[2] = 1.0f;
        specref[3] = 1.0f;

        // Here we initialize our multi-texturing functions
#ifdef SOFA_HAVE_GLEW
        glewInit();
        if (!GLEW_ARB_multitexture)
            std::cerr << "Error: GL_ARB_multitexture not supported\n";
#endif

        _clearBuffer = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
        _lightModelTwoSides = false;

        glDepthFunc(GL_LEQUAL);
        glClearDepth(1.0);
        glEnable(GL_NORMALIZE);

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

        // Set light model
        glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, lmodel_local);
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

        // Setup 'light 0'
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);

        // Enable color tracking
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // All materials hereafter have full specular reflectivity with a high shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
        glMateriali(GL_FRONT, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);

        // Define background color
        //	    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //Load texture for logo
        setBackgroundImage();

        glEnableClientState(GL_VERTEX_ARRAY);
        //glEnableClientState(GL_NORMAL_ARRAY);

        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

        //init Quadrics
        _arrow = gluNewQuadric();
        gluQuadricDrawStyle(_arrow, GLU_FILL);
        gluQuadricOrientation(_arrow, GLU_OUTSIDE);
        gluQuadricNormals(_arrow, GLU_SMOOTH);

        _tube = gluNewQuadric();
        gluQuadricDrawStyle(_tube, GLU_FILL);
        gluQuadricOrientation(_tube, GLU_OUTSIDE);
        gluQuadricNormals(_tube, GLU_SMOOTH);

        _sphere = gluNewQuadric();
        gluQuadricDrawStyle(_sphere, GLU_FILL);
        gluQuadricOrientation(_sphere, GLU_OUTSIDE);
        gluQuadricNormals(_sphere, GLU_SMOOTH);

        _disk = gluNewQuadric();
        gluQuadricDrawStyle(_disk, GLU_FILL);
        gluQuadricOrientation(_disk, GLU_OUTSIDE);
        gluQuadricNormals(_disk, GLU_SMOOTH);

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = CTime::getTime();

        printf("GL initialized\n");
    }



    // switch to preset view
    resetView();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::PrintString(void* font, char* string)
{
    int len, i;

    len = (int) strlen(string);
    for (i = 0; i < len; i++)
    {
        glutBitmapCharacter(font, string[i]);
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::Display3DText(float x, float y, float z, char* string)
{
    char* c;

    glPushMatrix();
    glTranslatef(x, y, z);
    for (c = string; *c != '\0'; c++)
    {
        glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
    }
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void QtViewer::DrawAxis(double xpos, double ypos, double zpos, double arrowSize)
{
    float fontScale = (float) (arrowSize / 600.0);

    Enable<GL_DEPTH_TEST> depth;
    Enable<GL_LIGHTING> lighting;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the "X" axis in red
    glPushMatrix();
    glColor3f(1.0, 0.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(90.0f, 0.0, 1.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "X" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double) glutStrokeWidth(GLUT_STROKE_ROMAN,
            88), arrowSize / 15.0, arrowSize / 5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 88);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0f);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Y" axis in green
    glPushMatrix();
    glColor3f(0.0, 1.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(-90.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Y" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double) glutStrokeWidth(GLUT_STROKE_ROMAN,
            89), arrowSize / 15.0, arrowSize / 5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 89);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Z" axis in blue
    glPushMatrix();
    glColor3f(0.0, 0.0, 1.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(0.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Z" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double) glutStrokeWidth(GLUT_STROKE_ROMAN,
            90), arrowSize / 15.0, arrowSize / 5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 90);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void QtViewer::DrawBox(SReal* minBBox, SReal* maxBBox, SReal r)
{
    //std::cout << "box = < " << minBBox[0] << ' ' << minBBox[1] << ' ' << minBBox[2] << " >-< " << maxBBox[0] << ' ' << maxBBox[1] << ' ' << maxBBox[2] << " >"<< std::endl;
    if (r == 0.0)
        r = (Vector3(maxBBox) - Vector3(minBBox)).norm() / 500;

    Enable<GL_DEPTH_TEST> depth;
    Enable<GL_LIGHTING> lighting;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the corners
    glColor3f(0.0, 1.0, 1.0);
    for (int corner = 0; corner < 8; ++corner)
    {
        glPushMatrix();
        glTranslated((corner & 1) ? minBBox[0] : maxBBox[0],
                (corner & 2) ? minBBox[1] : maxBBox[1],
                (corner & 4) ? minBBox[2] : maxBBox[2]);
        gluSphere(_sphere, 2 * r, 20, 10);
        glPopMatrix();
    }

    glColor3f(1.0, 1.0, 0.0);
    // --- Draw the X edges
    for (int corner = 0; corner < 4; ++corner)
    {
        glPushMatrix();
        glTranslated(minBBox[0], (corner & 1) ? minBBox[1] : maxBBox[1],
                (corner & 2) ? minBBox[2] : maxBBox[2]);
        glRotatef(90, 0, 1, 0);
        gluCylinder(_tube, r, r, maxBBox[0] - minBBox[0], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Y edges
    for (int corner = 0; corner < 4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner & 1) ? minBBox[0] : maxBBox[0], minBBox[1],
                (corner & 2) ? minBBox[2] : maxBBox[2]);
        glRotatef(-90, 1, 0, 0);
        gluCylinder(_tube, r, r, maxBBox[1] - minBBox[1], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Z edges
    for (int corner = 0; corner < 4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner & 1) ? minBBox[0] : maxBBox[0],
                (corner & 2) ? minBBox[1] : maxBBox[1], minBBox[2]);
        gluCylinder(_tube, r, r, maxBBox[2] - minBBox[2], 10, 10);
        glPopMatrix();
    }
}

// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void QtViewer::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
        double ymax, double step)
{
    register double x, y;

    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (x = xmin; x <= xmax; x += step)
    {
        glVertex3d(x, ymin, zo);
        glVertex3d(x, ymax, zo);
    }
    glEnd();

    glBegin(GL_LINES);
    for (y = ymin; y <= ymax; y += step)
    {
        glVertex3d(xmin, y, zo);
        glVertex3d(xmax, y, zo);
    }
    glEnd();
}

// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void QtViewer::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
        double zmax, double step)
{
    register double y, z;
    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (y = ymin; y <= ymax; y += step)
    {
        glVertex3d(xo, y, zmin);
        glVertex3d(xo, y, zmax);
    }
    glEnd();

    glBegin(GL_LINES);
    for (z = zmin; z <= zmax; z += step)
    {
        glVertex3d(xo, ymin, z);
        glVertex3d(xo, ymax, z);
    }
    glEnd();

}

// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void QtViewer::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
        double zmax, double step)
{
    register double x, z;
    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (x = xmin; x <= xmax; x += step)
    {
        glVertex3d(x, yo, zmin);
        glVertex3d(x, yo, zmax);
    }
    glEnd();

    glBegin(GL_LINES);
    for (z = zmin; z <= zmax; z += step)
    {
        glVertex3d(xmin, yo, z);
        glVertex3d(xmax, yo, z);
    }
    glEnd();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtViewer::DrawLogo()
{
    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage())
    {
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();
    }
    else
        return;

    Enable<GL_TEXTURE_2D> tex;
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (texLogo)
        texLogo->bind();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d((_W - w) / 2, (_H - h) / 2, 0.0);

    glTexCoord2d(1.0, 0.0);
    glVertex3d(_W - (_W - w) / 2, (_H - h) / 2, 0.0);

    glTexCoord2d(1.0, 1.0);
    glVertex3d(_W - (_W - w) / 2, _H - (_H - h) / 2, 0.0);

    glTexCoord2d(0.0, 1.0);
    glVertex3d((_W - w) / 2, _H - (_H - h) / 2, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}



// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::drawColourPicking(core::CollisionModel::ColourCode code)
{
    // Define background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(lastProjectionMatrix);
    glMatrixMode(GL_MODELVIEW);


    sofa::simulation::ColourPickingVisitor cpv(sofa::core::ExecParams::defaultInstance(), code);
    cpv.execute(sofa::simulation::getSimulation()->getContext() );

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

}
// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtViewer::DisplayOBJs()
{
    if (!groot)
        return;
    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1, 1, 1, 1);
    glDisable(GL_COLOR_MATERIAL);

    if (!initTexturesDone)
    {
        // 		std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        getSimulation()->initTextures(groot);
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {

        getSimulation()->draw(groot, &visualParameters);
        getSimulation()->draw(simulation::getSimulation()->getVisualRoot(),
                &visualParameters);
        if (_axis)
        {
            DrawAxis(0.0, 0.0, 0.0, 10.0);
            if (visualParameters.minBBox[0] < visualParameters.maxBBox[0])
                DrawBox(visualParameters.minBBox.ptr(),
                        visualParameters.maxBBox.ptr());
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void QtViewer::DisplayMenu(void)
{
    Disable<GL_LIGHTING> light;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.3f, 0.7f, 0.95f);
    glRasterPos2i(_W / 2 - 5, _H - 15);
    //sprintf(buffer,"FPS: %.1f\n", _frameRate.GetFPS());
    //PrintString(GLUT_BITMAP_HELVETICA_12, buffer);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void QtViewer::MakeStencilMask()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0,_W, 0, _H );
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glClear(GL_STENCIL_BUFFER_BIT);
    glStencilFunc(GL_ALWAYS, 0x1, 0x1);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    glColor4f(0,0,0,0);
    glBegin(GL_LINES);
    for (float f=0 ; f< _H ; f+=2.0)
    {
        glVertex2f(0.0, f);
        glVertex2f(_W, f);
    }
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::DrawScene(void)
{
    if(!currentCamera)
        return ;

    calcProjection();

    if (_background == 0)
        DrawLogo();

    glLoadIdentity();

    //visualParameters.sceneTransform.Apply();
    GLdouble mat[16];
    //currentCamera->getOpenGLModelViewMatrix(mat);
    currentCamera->getOpenGLMatrix(mat);
    glMultMatrixd(mat);

    glGetDoublev(GL_MODELVIEW_MATRIX, lastModelviewMatrix);

    //for(int i=0 ; i<16 ;i++)
    //	std::cout << lastModelviewMatrix[i] << " ";
//
//	std::cout << std::endl;

    //Vec position() const { return inverseCoordinatesOf(Vec(0.0,0.0,0.0)); };

    //std::cout << "P " << currentCamera->getPosition() << std::endl;


    if(currentCamera)
    {
        //	std::cout << currentCamera->getPosition() << " " << currentCamera->getOrientation() << std::endl;
        //	std::cout << currentCamera->getZNear() << " " << currentCamera->getZFar() << std::endl;
    }

    if (_renderingMode == GL_RENDER)
    {
        //STEREO MODE
        if(_stereoEnabled)
        {
            //calcProjection();

            //window()->showNormal();
            glEnable(GL_STENCIL_TEST);
            MakeStencilMask();

            //1st pass
            glStencilFunc(GL_EQUAL, 0x1, 0x1);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            DisplayOBJs();

            //2nd pass
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            //glLoadIdentity();
            //translate slighty the camera
            visualParameters.sceneTransform.translation[0] += _stereoShift;
            visualParameters.sceneTransform.Apply();

            glStencilFunc(GL_NOTEQUAL, 0x1, 0x1);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            DisplayOBJs();
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
            glDisable(GL_STENCIL_TEST);

            visualParameters.sceneTransform.translation[0] -= _stereoShift;
        }
        else
        {
            //SPLIT MODE
            if (_binocularModeEnabled)
            {
                glMatrixMode(GL_PROJECTION);
                glPushMatrix();
                glViewport(0, 0, _W/2, _H);
                glPopMatrix();
                glMatrixMode(GL_MODELVIEW);
                DisplayOBJs();

                glMatrixMode(GL_PROJECTION);
                glPushMatrix();
                glViewport(_W/2, 0, _W, _H);
                glPopMatrix();
                glMatrixMode(GL_MODELVIEW);
                DisplayOBJs();
            }
            //NORMAL MODE
            else
            {
                //calcProjection(0,0, _W, _H);
                //window()->showNormal();
                DisplayOBJs();
            }
        }

        DisplayMenu(); // always needs to be the last object being drawn
    }

}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void QtViewer::resizeGL(int width, int height)
{

    _W = width;
    _H = height;

    if(currentCamera)
        currentCamera->setViewport(width, height);

    // 	std::cout << "GL window: " <<width<<"x"<<height <<std::endl;

    calcProjection();
    this->resize(width, height);
    emit( resizeW(_W));
    emit( resizeH(_H));
}

// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void QtViewer::calcProjection()
{
    int width = _W;
    int height = _H;
    double xNear, yNear, xOrtho, yOrtho;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground,
           zBackground;
    Vector3 center;

    /// Camera part
    if (!currentCamera)
        return;
    /*
    	const Vec3d& camPosition = currentCamera->getPosition();
    	const Quat& camOrientation = currentCamera->getOrientation();

    	//std::cout << camPosition << std::endl;

    	visualParameters.sceneTransform.translation[0] = -camPosition[0];
    	visualParameters.sceneTransform.translation[1] = -camPosition[1];
    	visualParameters.sceneTransform.translation[2] = -camPosition[2];

    	camOrientation.buildRotationMatrix(visualParameters.sceneTransform.rotation);
    */
    if (groot && (!sceneBBoxIsValid || _axis))
    {
        getSimulation()->computeBBox(groot, visualParameters.minBBox.ptr(),
                visualParameters.maxBBox.ptr());
        getSimulation()->computeBBox(getSimulation()->getVisualRoot(),
                visualParameters.minBBox.ptr(), visualParameters.maxBBox.ptr(),
                false);
        sceneBBoxIsValid = true;

        currentCamera->setBoundingBox(visualParameters.minBBox, visualParameters.maxBBox);
    }

    visualParameters.zNear = currentCamera->getZNear();
    visualParameters.zFar = currentCamera->getZFar();
    ///


    xNear = 0.35 * visualParameters.zNear;
    yNear = 0.35 * visualParameters.zNear;
    offset = 0.001 * visualParameters.zNear; // for foreground and background planes

    xOrtho = fabs(visualParameters.sceneTransform.translation[2]) * xNear
            / visualParameters.zNear;
    yOrtho = fabs(visualParameters.sceneTransform.translation[2]) * yNear
            / visualParameters.zNear;

    if ((height != 0) && (width != 0))
    {
        if (height > width)
        {
            xFactor = 1.0;
            yFactor = (double) height / (double) width;
        }
        else
        {
            xFactor = (double) width / (double) height;
            yFactor = 1.0;
        }
    }

    visualParameters.viewport[0] = 0;
    visualParameters.viewport[1] = 0;
    visualParameters.viewport[2] = width;
    visualParameters.viewport[3] = height;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    //std::cout << xNear << " " << yNear << std::endl;

    zForeground = -visualParameters.zNear - offset;
    zBackground = -visualParameters.zFar + offset;

    if (currentCamera->getCameraType() == component::visualmodel::BaseCamera::PERSPECTIVE_TYPE)
        gluPerspective(currentCamera->getFieldOfView(), (double) width / (double) height, visualParameters.zNear, visualParameters.zFar);
    else
    {
        float ratio = visualParameters.zFar / (visualParameters.zNear * 20);
        Vector3 tcenter = visualParameters.sceneTransform * center;
        if (tcenter[2] < 0.0)
        {
            ratio = -300 * (tcenter.norm2()) / tcenter[2];
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear
                * yFactor) * ratio, (yNear * yFactor) * ratio,
                visualParameters.zNear, visualParameters.zFar);
    }

    xForeground = -zForeground * xNear / visualParameters.zNear;
    yForeground = -zForeground * yNear / visualParameters.zNear;
    xBackground = -zBackground * xNear / visualParameters.zNear;
    yBackground = -zBackground * yNear / visualParameters.zNear;

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glGetDoublev(GL_PROJECTION_MATRIX, lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::paintGL()
{

    // clear buffers (color and depth)
    if (_background == 0)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    else if (_background == 1)
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    else if (_background == 2)
        glClearColor(backgroundColour[0], backgroundColour[1],
                backgroundColour[2], 1.0f);
    glClearDepth(1.0);
    glClear( _clearBuffer);

    // draw the scene
    DrawScene();

    if (_video)
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ((counter++ % CAPTURE_PERIOD)==0)
#endif
        }

    SofaViewer::captureEvent();

    if (_waitForRender)
        _waitForRender = false;

    emit( redrawn());
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::ApplySceneTransformation(int /* x */, int /* y */)
{
    update();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::ApplyMouseInteractorTransformation(int x, int y)
{
    // Mouse Interaction
    double coeffDeplacement = 0.025;
    if (sceneBBoxIsValid && visualParameters.maxBBox[0]
        > visualParameters.minBBox[0])
        coeffDeplacement *= 0.001 * (visualParameters.maxBBox
                - visualParameters.minBBox).norm();
    Quaternion conjQuat, resQuat, _newQuatBckUp;

    float x1, x2, y1, y2;

    if (_mouseInteractorMoving)
    {
        if (_mouseInteractorRotationMode)
        {
            if ((_mouseInteractorSavedPosX != x) || (_mouseInteractorSavedPosY
                    != y))
            {
                x1 = 0;
                y1 = 0;
                x2 = (2.0f * (x + (-_mouseInteractorSavedPosX + _W / 2.0f))
                        - _W) / _W;
                y2 = (_H - 2.0f
                        * (y + (-_mouseInteractorSavedPosY + _H / 2.0f))) / _H;

                _mouseInteractorTrackball.ComputeQuaternion(x1, y1, x2, y2);
                _mouseInteractorCurrentQuat
                    = _mouseInteractorTrackball.GetQuaternion();
                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;

                _mouseInteractorNewQuat = _mouseInteractorCurrentQuat
                        + _mouseInteractorNewQuat;
                _mouseRotate = true;
            }
            else
            {
                _mouseRotate = false;
            }

            update();
        }
        else if (_mouseInteractorTranslationMode)
        {
            _mouseInteractorAbsolutePosition = Vector3(0, 0, 0);
            _mouseInteractorRelativePosition = Vector3(0, 0, 0);

            if (_translationMode == XY_TRANSLATION)
            {
                _mouseInteractorAbsolutePosition[0] = coeffDeplacement * (x
                        - _mouseInteractorSavedPosX);
                _mouseInteractorAbsolutePosition[1] = -coeffDeplacement * (y
                        - _mouseInteractorSavedPosY);

                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;
            }
            else if (_translationMode == Z_TRANSLATION)
            {
                _mouseInteractorAbsolutePosition[2] = coeffDeplacement * (y
                        - _mouseInteractorSavedPosY);

                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;
            }

            _newQuatBckUp[0] = _newQuat[0];
            _newQuatBckUp[1] = _newQuat[1];
            _newQuatBckUp[2] = _newQuat[2];
            _newQuatBckUp[3] = _newQuat[3];

            _newQuatBckUp.normalize();

            // Conjugate calculation of the scene orientation quaternion
            conjQuat[0] = -_newQuatBckUp[0];
            conjQuat[1] = -_newQuatBckUp[1];
            conjQuat[2] = -_newQuatBckUp[2];
            conjQuat[3] = _newQuatBckUp[3];

            conjQuat.normalize();

            resQuat = _newQuatBckUp.quatVectMult(
                    _mouseInteractorAbsolutePosition) * conjQuat;

            _mouseInteractorRelativePosition[0] = resQuat[0];
            _mouseInteractorRelativePosition[1] = resQuat[1];
            _mouseInteractorRelativePosition[2] = resQuat[2];

            _mouseTrans = true;
            update();
        }
    }
}

// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------


void QtViewer::keyPressEvent(QKeyEvent * e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->key() == Qt::Key_Escape)
        {
            m_grabActived = false;
            this->setCursor(QCursor(Qt::ArrowCursor));
        }
        else
        {
            if (groot)
            {
                sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
                groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
            }
            return;
        }
    }
#endif
    if (isControlPressed()) // pass event to the scene data structure
    {
#ifdef TRACKING_MOUSE
        if (e->key() == Qt::Key_T)
        {
            m_grabActived = true;
            this->setCursor(QCursor(Qt::BlankCursor));
            QPoint p = mapToGlobal(this->pos()) + QPoint((this->width()+2)/2,(this->height()+2)/2);
            QCursor::setPos(p);
        }
        else
#endif
            //	cerr<<"QtViewer::keyPressEvent, key = "<<e->key()<<" with Control pressed "<<endl;
            if (groot)
            {
                sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
                groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
            }
    }
    else
        // control the GUI
        switch (e->key())
        {

#ifdef TRACKING
        case Qt::Key_X:
        {
            tracking = !tracking;
            break;
        }
#endif // TRACKING
        case Qt::Key_C:
        {
            // --- switch interaction mode
            if (!_mouseInteractorTranslationMode)
            {
                std::cout << "Interaction Mode ON\n";
                _mouseInteractorTranslationMode = true;
                _mouseInteractorRotationMode = false;
            }
            else
            {
                std::cout << "Interaction Mode OFF\n";
                _mouseInteractorTranslationMode = false;
                _mouseInteractorRotationMode = false;
            }
            break;
        }
        default:
        {
            SofaViewer::keyPressEvent(e);
            e->ignore();
        }
        update();
        }
}

void QtViewer::keyReleaseEvent(QKeyEvent * e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
        if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        return;
    }
#endif
    SofaViewer::keyReleaseEvent(e);
}

void QtViewer::wheelEvent(QWheelEvent* e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
        if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        return;
    }
#endif
    if (e->state() & Qt::ControlButton)
    {
        moveLaparoscopic(e);
    }

    SofaViewer::wheelEvent(e);
}

void QtViewer::mousePressEvent(QMouseEvent * e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->type() == QEvent::MouseButtonPress)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed);
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed);
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
#endif
    mouseEvent(e);

    SofaViewer::mousePressEvent(e);
}

void QtViewer::mouseReleaseEvent(QMouseEvent * e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->type() == QEvent::MouseButtonRelease)
        {
            if (e->button() == Qt::LeftButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
#endif
    mouseEvent(e);

    SofaViewer::mouseReleaseEvent(e);

}

void QtViewer::mouseMoveEvent(QMouseEvent * e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        QPoint p = mapToGlobal(this->pos()) + QPoint((this->width()+2)/2,(this->height()+2)/2);
        QPoint c = QCursor::pos();
        sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move,c.x() - p.x(),c.y() - p.y());
        QCursor::setPos(p);
        if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        return;
    }
#endif

#ifdef TRACKING
    if (tracking)
    {
        if (groot)
        {
            if (firstTime)
            {
                savedX = e->x();
                savedY = e->y();
                firstTime = false;
            }

            sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move,e->x()-savedX,e->y()-savedY);
            groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            QCursor::setPos(mapToGlobal(QPoint(savedX, savedY)));
        }
    }
    else
    {
        firstTime = true;
    }
#endif // TRACKING
    //if the mouse move is not "interactive", give the event to the camera
    if(!mouseEvent(e))
        SofaViewer::mouseMoveEvent(e);
}

// ---------------------- Here are the mouse controls for the scene  ----------------------
bool QtViewer::mouseEvent(QMouseEvent * e)
{
    bool isInteractive = false;
    int eventX = e->x();
    int eventY = e->y();
    if (_mouseInteractorRotationMode)
    {
        switch (e->type())
        {
        case QEvent::MouseButtonPress:
            // Mouse left button is pushed
            if (e->button() == Qt::LeftButton)
            {
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            break;

        case QEvent::MouseMove:
            //
            break;

        case QEvent::MouseButtonRelease:
            // Mouse left button is released
            if (e->button() == Qt::LeftButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            break;

        default:
            break;
        }
        ApplyMouseInteractorTransformation(eventX, eventY);
    }
    else if (_mouseInteractorTranslationMode)
    {
        switch (e->type())
        {
        case QEvent::MouseButtonPress:
            // Mouse left button is pushed
            if (e->button() == Qt::LeftButton)
            {
                _translationMode = XY_TRANSLATION;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
                _mouseInteractorMoving = true;
            }
            // Mouse right button is pushed
            else if (e->button() == Qt::RightButton)
            {
                _translationMode = Z_TRANSLATION;
                _mouseInteractorSavedPosY = eventY;
                _mouseInteractorMoving = true;
            }

            break;

        case QEvent::MouseButtonRelease:
            // Mouse left button is released
            if ((e->button() == Qt::LeftButton) && (_translationMode
                    == XY_TRANSLATION))
            {
                _mouseInteractorMoving = false;
            }
            // Mouse right button is released
            else if ((e->button() == Qt::RightButton) && (_translationMode
                    == Z_TRANSLATION))
            {
                _mouseInteractorMoving = false;
            }
            break;

        default:
            break;
        }

        ApplyMouseInteractorTransformation(eventX, eventY);
    }
    else if (e->state() & Qt::ShiftButton)
    {
        isInteractive = true;
        SofaViewer::mouseEvent(e);
    }
    else if (e->state() & Qt::ControlButton)
    {
        isInteractive = true;
        moveLaparoscopic(e);
    }
    else if (e->state() & Qt::AltButton)
    {
        isInteractive = true;
        switch (e->type())
        {
        case QEvent::MouseButtonPress:
            // Mouse left button is pushed
            if (e->button() == Qt::LeftButton)
            {
                _navigationMode = BTLEFT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse right button is pushed
            else if (e->button() == Qt::RightButton)
            {
                _navigationMode = BTRIGHT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse middle button is pushed
            else if (e->button() == Qt::MidButton)
            {
                _navigationMode = BTMIDDLE_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            break;

        case QEvent::MouseMove:
            //
            break;

        case QEvent::MouseButtonRelease:
            // Mouse left button is released
            if (e->button() == Qt::LeftButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse right button is released
            else if (e->button() == Qt::RightButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse middle button is released
            else if (e->button() == Qt::MidButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            break;

        default:
            break;
        }
        if (_mouseInteractorMoving && _navigationMode == BTLEFT_MODE)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                _lightPosition[0] -= dx * 0.1;
                _lightPosition[1] += dy * 0.1;
                std::cout << "Light = " << _lightPosition[0] << " "
                        << _lightPosition[1] << " " << _lightPosition[2]
                        << std::endl;
                update();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
        else if (_mouseInteractorMoving && _navigationMode == BTRIGHT_MODE)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                //g_DepthBias[0] += dx*0.01;
                g_DepthBias[1] += dy * 0.01;
                std::cout << "Depth bias = " << g_DepthBias[0] << " "
                        << g_DepthBias[1] << std::endl;
                update();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
        else if (_mouseInteractorMoving && _navigationMode == BTMIDDLE_MODE)
        {

            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                g_DepthOffset[0] += dx * 0.01;
                g_DepthOffset[1] += dy * 0.01;
                std::cout << "Depth offset = " << g_DepthOffset[0] << " "
                        << g_DepthOffset[1] << std::endl;
                update();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
    }

    return isInteractive;
}

void QtViewer::moveRayPickInteractor(int eventX, int eventY)
{

    Vec3d p0, px, py, pz, px1, py1;
    gluUnProject(eventX, visualParameters.viewport[3] - 1 - (eventY), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(p0[0]), &(p0[1]), &(p0[2]));
    gluUnProject(eventX + 1, visualParameters.viewport[3] - 1 - (eventY), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(px[0]), &(px[1]), &(px[2]));
    gluUnProject(eventX, visualParameters.viewport[3] - 1 - (eventY + 1), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(py[0]), &(py[1]), &(py[2]));
    gluUnProject(eventX, visualParameters.viewport[3] - 1 - (eventY), 0.1,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(pz[0]), &(pz[1]), &(pz[2]));
    gluUnProject(eventX + 1, visualParameters.viewport[3] - 1 - (eventY), 0.1,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(px1[0]), &(px1[1]), &(px1[2]));
    gluUnProject(eventX, visualParameters.viewport[3] - 1 - (eventY + 1), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            visualParameters.viewport, &(py1[0]), &(py1[1]), &(py1[2]));
    px1 -= pz;
    py1 -= pz;
    px -= p0;
    py -= p0;
    pz -= p0;
    double r0 = sqrt(px.norm2() + py.norm2());
    double r1 = sqrt(px1.norm2() + py1.norm2());
    r1 = r0 + (r1 - r0) / pz.norm();
    px.normalize();
    py.normalize();
    pz.normalize();
    Mat4x4d transform;
    transform.identity();
    transform[0][0] = px[0];
    transform[1][0] = px[1];
    transform[2][0] = px[2];
    transform[0][1] = py[0];
    transform[1][1] = py[1];
    transform[2][1] = py[2];
    transform[0][2] = pz[0];
    transform[1][2] = pz[1];
    transform[2][2] = pz[2];
    transform[0][3] = p0[0];
    transform[1][3] = p0[1];
    transform[2][3] = p0[2];
    Mat3x3d mat;
    mat = transform;
    Quat q;
    q.fromMatrix(mat);

    Vec3d position, direction;
    position = transform * Vec4d(0, 0, 0, 1);
    direction = transform * Vec4d(0, 0, 1, 0);
    direction.normalize();
    pick.updateRay(position, direction);
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtViewer::resetView()
{
    Vec3d position;
    Quat orientation;
    bool fileRead = false;

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + "." + VIEW_FILE_EXTENSION;
        /*std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
        	in >> position[0];
        	in >> position[1];
        	in >> position[2];
        	in >> orientation[0];
        	in >> orientation[1];
        	in >> orientation[2];
        	in >> orientation[3];
        	orientation.normalize();

        	in.close();
        	fileRead = true;

            setView(position, orientation);
        }*/
        fileRead = currentCamera->importParametersFromFile(viewFileName);
    }

    //if there is no .view file , look at the center of the scene bounding box
    // and with a Up vector in the same axis as the gravity
    if (!fileRead)
    {
        newView();
    }

    update();
    //updateGL();

    //SofaViewer::resetView();
    //ResetScene();
}

void QtViewer::newView()
{
    SofaViewer::newView();
}

void QtViewer::getView(Vec3d& pos, Quat& ori) const
{
    SofaViewer::getView(pos, ori);
}

void QtViewer::setView(const Vec3d& pos, const Quat &ori)
{
    SofaViewer::setView(pos, ori);
}

void QtViewer::moveView(const Vec3d& pos, const Quat &ori)
{
    SofaViewer::moveView(pos, ori);
}

void QtViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + "." + VIEW_FILE_EXTENSION;
        /*std::ofstream out(viewFileName.c_str());
        if (!out.fail())
        {
            const Vec3d& camPosition = currentCamera->getPosition();
        	const Quat& camOrientation = currentCamera->getOrientation();

        	out << camPosition[0] << " "
        		<< camPosition[1] << " "
        		<< camPosition[2] << "\n";
        	out << camOrientation[0] << " "
        	    << camOrientation[1] << " "
        	    << camOrientation[2] << " "
        	    << camOrientation[3] << "\n";
        	out.close();
        }*/
        if(currentCamera->exportParametersInFile(viewFileName))
            std::cout << "View parameters saved in " << viewFileName << std::endl;
        else
            std::cout << "Error while saving view parameters in " << viewFileName << std::endl;
    }
}

void QtViewer::setScene(sofa::simulation::Node* scene, const char* filename,
        bool keepParams)
{

    bool newScene = (scene != groot);
    SofaViewer::setScene(scene, filename, keepParams);
    if (newScene)
    {
        getSimulation()->computeBBox(groot, visualParameters.minBBox.ptr(),
                visualParameters.maxBBox.ptr());
        getSimulation()->computeBBox(getSimulation()->getVisualRoot(),
                visualParameters.minBBox.ptr(), visualParameters.maxBBox.ptr(),
                false);
    }
}

void QtViewer::setSizeW(int size)
{
    resizeGL(size, _H);
    updateGL();
}

void QtViewer::setSizeH(int size)
{
    resizeGL(_W, size);
    updateGL();
}

QString QtViewer::helpString()
{
    QString
    text(
        "<H1>QtViewer</H1><hr>\
<ul>\
<li><b>Mouse</b>: TO NAVIGATE<br></li>\
<li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
<li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
<li><b>C</b>: TO SWITCH INTERACTION MODE: press the KEY C.<br>\
Allow or not the navigation with the mouse.<br></li>\
<li><b>Ctrl + L</b>: TO DRAW SHADOWS<br></li>\
<li><b>O</b>: TO EXPORT TO .OBJ<br>\
The generated files scene-time.obj and scene-time.mtl are saved in the running project directory<br></li>\
<li><b>P</b>: TO SAVE A SEQUENCE OF OBJ<br>\
Each time the frame is updated an obj is exported<br></li>\
<li><b>R</b>: TO DRAW THE SCENE AXIS<br></li>\
<li><b>S</b>: TO SAVE A SCREENSHOT<br>\
The captured images are saved in the running project directory under the name format capturexxxx.bmp<br></li>\
<li><b>T</b>: TO CHANGE BETWEEN A PERSPECTIVE OR AN ORTHOGRAPHIC CAMERA<br></li>\
<li><b>V</b>: TO SAVE A VIDEO<br>\
Each time the frame is updated a screenshot is saved<br></li>\
<li><b>Esc</b>: TO QUIT ::sofa:: <br></li></ul>");
    return text;
}

}// namespace qt

} // namespace viewer

}

} // namespace gui

} // namespace sofa
