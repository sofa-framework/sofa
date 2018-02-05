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
#include <sofa/gui/qt/viewer/qt/QtViewer.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <qevent.h>

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#endif

#include <sofa/gui/qt/GenGraphForm.h>


#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/RAII.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/gui/ColourPickingVisitor.h>

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

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
QSurfaceFormat QtViewer::setupGLFormat(const unsigned int nbMSAASamples)
{
    QSurfaceFormat f = QSurfaceFormat::defaultFormat();

    //Multisampling
    if(nbMSAASamples > 1)
    {
        std::cout <<"QtViewer: Set multisampling anti-aliasing (MSSA) with " << nbMSAASamples << " samples." << std::endl;
        f.setSamples(nbMSAASamples);
    }

    //VSync
    std::cout << "QtViewer: disabling vertical refresh sync" << std::endl;
    f.setSwapInterval(0); // disable vertical refresh sync

    int vmajor = 3, vminor = 2;
    f.setVersion(vmajor,vminor);
    f.setProfile(QSurfaceFormat::CompatibilityProfile);

    f.setSwapBehavior(QSurfaceFormat::DoubleBuffer);

    return f;
}
#else
QGLFormat QtViewer::setupGLFormat(const unsigned int nbMSAASamples)
{
    QGLFormat f = QGLFormat::defaultFormat();

    if(nbMSAASamples > 1)
    {
        std::cout <<"QtViewer: Set multisampling anti-aliasing (MSSA) with " << nbMSAASamples << " samples." << std::endl;
        f.setSampleBuffers(true);
        f.setSamples(nbMSAASamples);
    }

//    int val = 0;

#if defined(QT_VERSION) && QT_VERSION >= 0x040200
    std::cout << "QtViewer: disabling vertical refresh sync" << std::endl;
    f.setSwapInterval(0); // disable vertical refresh sync
#endif
#if defined(QT_VERSION) && QT_VERSION >= 0x040700
    int vmajor = 3, vminor = 2;
    //int vmajor = 4, vminor = 2;
    //std::cout << "QtViewer: Trying to open an OpenGL " << vmajor << "." << vminor << " compatibility profile context" << std::endl;
    f.setVersion(vmajor,vminor);
    f.setProfile(QGLFormat::CompatibilityProfile);
#endif
    //f.setOption(QGL::SampleBuffers);
    return f;
}

#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400

// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
QtViewer::QtViewer(QWidget* parent, const char* name, const unsigned int nbMSAASamples)
#if defined(QT_VERSION) && QT_VERSION >= 0x050400
    : QOpenGLWidget(parent)
 #else
    : QOpenGLWidget(setupGLFormat(nbMSAASamples), parent)
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400
{
#ifdef __linux__
    ::setenv("MESA_GL_VERSION_OVERRIDE", "3.0", 1);
#endif // __linux

    this->setObjectName(name);

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
    this->setFormat(setupGLFormat(nbMSAASamples));
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400

    groot = NULL;
    initTexturesDone = false;
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
    std::cout << "QtViewer: OpenGL " << glGetString(GL_VERSION)
              << " context created." << std::endl;
    if (std::string((const char*)glGetString(GL_VENDOR)).find("Intel") !=
            std::string::npos)
    {
        const char* mesaEnv = ::getenv("MESA_GL_VERSION_OVERRIDE");
        if ( !mesaEnv || std::string(mesaEnv) != "3.0")
            msg_error("runSofa") << "QtViewer is not compatible with Intel drivers on "
                                    "Linux. To use runSofa, either change the gui to "
                                    "qglviewer (runSofa -g qglviewer) or set the "
                                    "environment variable \"MESA_GL_VERSION_OVERRIDE\" "
                                    "to the value \"3.0\"";
    }

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
            msg_error("QtViewer") << "GL_ARB_multitexture not supported.";
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

        //printf("GL initialized\n");
    }

    // switch to preset view
    resetView();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::PrintString(void* /*font*/, char* string)
{
    helper::gl::GlText::draw(string);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::Display3DText(float x, float y, float z, char* string)
{
    glPushMatrix();
    glTranslatef(x, y, z);
    helper::gl::GlText::draw(string);
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void QtViewer::DrawAxis(double xpos, double ypos, double zpos, double arrowSize)
{
    float fontScale = (float) (arrowSize * 0.25);

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
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);

    helper::gl::GlText::draw('X', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);

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
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);
    helper::gl::GlText::draw('Y', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);
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
    glTranslated(-0.5 * fontScale, arrowSize / 15.0, arrowSize / 5.0);
    helper::gl::GlText::draw('Z', sofa::defaulttype::Vector3(0.0, 0.0, 0.0), fontScale);
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
    double x, y;

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
    double y, z;
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
    double x, z;
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
    glPushMatrix();
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
    glPopMatrix();
}



// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtViewer::drawColourPicking(ColourPickingVisitor::ColourCode code)
{
    // Define background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(lastProjectionMatrix);
    glMatrixMode(GL_MODELVIEW);


    ColourPickingVisitor cpv(sofa::core::visual::VisualParams::defaultInstance(), code);
    cpv.execute( groot.get() );

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

    if (_background == 0)
        DrawLogo();

    if (!groot)
        return;
    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;


    vparams->sceneBBox() = groot->f_bbox.getValue();


    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1, 1, 1, 1);
    glDisable(GL_COLOR_MATERIAL);

    if (!initTexturesDone)
    {
        // 		std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        getSimulation()->initTextures(groot.get());
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {

        getSimulation()->draw(vparams,groot.get());

        if (_axis)
        {

            SReal* minBBox = vparams->sceneBBox().minBBoxPtr();
            SReal* maxBBox = vparams->sceneBBox().maxBBoxPtr();
            SReal maxDistance = std::numeric_limits<SReal>::min();

            maxDistance = maxBBox[0] - minBBox[0];
            for (int i=1;i<3;i++)
            {
                if(maxDistance < (maxBBox[i] - minBBox[i]))
                    maxDistance = (maxBBox[i] - minBBox[i]);
            }

            if(maxDistance == 0 )
                maxDistance = 1.0;

            // World Axis: Arrows of axis are defined as 10% of maxBBox
            DrawAxis(0.0, 0.0, 0.0,(maxDistance*0.1));

            if (vparams->sceneBBox().minBBox().x() < vparams->sceneBBox().maxBBox().x())
                DrawBox(vparams->sceneBBox().minBBoxPtr(), vparams->sceneBBox().maxBBoxPtr());

            // 2D Axis: project current world orientation in the lower left part of the screen
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0.0,vparams->viewport()[2],0,vparams->viewport()[3],-30,30);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            helper::gl::Axis::draw(sofa::defaulttype::Vector3(30.0,30.0,0.0),currentCamera->getOrientation().inverse(), 25.0);
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();

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
void QtViewer::drawScene(void)
{
    if (!groot) return;

    if(!currentCamera)
    {
        msg_error("QtViewer") << "No camera defined.";
        return;
    }

    int width = _W;
    int height = _H;
    bool stereo = currentCamera->getStereoEnabled();
    bool twopass = stereo;
    sofa::component::visualmodel::BaseCamera::StereoMode smode = currentCamera->getStereoMode();
    sofa::component::visualmodel::BaseCamera::StereoStrategy sStrat = currentCamera->getStereoStrategy();
    double sShift = currentCamera->getStereoShift();
    bool stencil = false;
    bool viewport = false;
    bool supportStereo = currentCamera->isStereo();
    sofa::core::visual::VisualParams::Viewport vpleft, vpright;
    if(supportStereo)
    {
        if(stereo)
        {
            if (smode == sofa::component::visualmodel::BaseCamera::STEREO_AUTO)
        {
            // auto-detect stereo mode
            static int prevsmode = sofa::component::visualmodel::BaseCamera::STEREO_AUTO;
            if ((_W <= 1280 && _H == 1470) || (_W <= 1920 && _H == 2205))
            {
                // standard HDMI 1.4 stereo frame packing format
                smode = sofa::component::visualmodel::BaseCamera::STEREO_FRAME_PACKING;
                if (smode != prevsmode) std::cout << "AUTO Stereo mode: Frame Packing" << std::endl;
            }
            else if (_W >= 2 * _H)
            {
                smode = sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE;
                if (smode != prevsmode) std::cout << "AUTO Stereo mode: Side by Side" << std::endl;
            }
            else if (_H > _W)
            {
                smode = sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM;
                if (smode != prevsmode) std::cout << "AUTO Stereo mode: Top Bottom" << std::endl;
            }
            else
            {
                smode = sofa::component::visualmodel::BaseCamera::STEREO_INTERLACED;
                if (smode != prevsmode) std::cout << "AUTO Stereo mode: Interlaced" << std::endl;
                //smode = STEREO_SYDE_BY_SIDE_HALF;
                //if (smode != prevsmode) std::cout << "AUTO Stereo mode: Side by Side Half" << std::endl;
            }
            prevsmode = smode;
        }
            switch (smode)
            {
            case sofa::component::visualmodel::BaseCamera::STEREO_INTERLACED:
            {
                stencil = true;
                glEnable(GL_STENCIL_TEST);
                MakeStencilMask();
                break;
            }
            case sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE:
            case sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE_HALF:
            {
                width /= 2;
                viewport = true;
                vpleft = sofa::helper::make_array(0,0,width,height);
                vpright = sofa::helper::make_array(_W-width,0,width,height);
                if (smode == sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE_HALF)
                    width = _W; // keep the original ratio for camera
                break;
            }
            case sofa::component::visualmodel::BaseCamera::STEREO_FRAME_PACKING:
            case sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM:
            case sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM_HALF:
            {
                if (smode == sofa::component::visualmodel::BaseCamera::STEREO_FRAME_PACKING && _H == 1470) // 720p format
                    height = 720;
                else if (smode == sofa::component::visualmodel::BaseCamera::STEREO_FRAME_PACKING && _H == 2205) // 1080p format
                    height = 1080;
                else // other resolutions
                    height /= 2;
                viewport = true;
                vpleft = sofa::helper::make_array(0,0,width,height);
                vpright = sofa::helper::make_array(0,_H-height,width,height);
                if (smode == sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM_HALF)
                    height = _H; // keep the original ratio for camera
                break;
            }
            case sofa::component::visualmodel::BaseCamera::STEREO_AUTO:
            case sofa::component::visualmodel::BaseCamera::STEREO_NONE:
            default:
                twopass = false;
                break;
            }
        }
    }else
    {
        twopass = false;
    }
    calcProjection(width, height);

    glLoadIdentity();

    GLdouble mat[16];

    //std::cout << "Default" << this->defaultFramebufferObject() << std::endl;
    currentCamera->getOpenGLModelViewMatrix(mat);
    glMultMatrixd(mat);

    glGetDoublev(GL_MODELVIEW_MATRIX, lastModelviewMatrix);
    vparams->setModelViewMatrix(lastModelviewMatrix);

    if(supportStereo)
    {
        if (stereo)
        {
            //1st pass
            if (viewport)
            {
                sofa::core::visual::VisualParams::Viewport vp = vpleft;
                vparams->viewport() = vp;
                glViewport(vp[0], vp[1], vp[2], vp[3]);
                glScissor(vp[0], vp[1], vp[2], vp[3]);
                glEnable(GL_SCISSOR_TEST);
            }
            if (stencil)
            {
                glStencilFunc(GL_EQUAL, 0x1, 0x1);
                glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
            }
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            if(sStrat == sofa::component::visualmodel::BaseCamera::PARALLEL)
            {
                glTranslated(sShift/2,0,0);
            }
            else if(sStrat == sofa::component::visualmodel::BaseCamera::TOEDIN)
            {
                double distance = currentCamera ? currentCamera->getDistance() : 10*sShift;
                double angle = atan2(sShift,distance)*180.0/M_PI;
                glTranslated(0,0,-distance);
                glRotated(-angle,0,1,0);
                glTranslated(0,0,distance);
            }


            glMultMatrixd(mat);
        }
    }

    if (_renderingMode == GL_RENDER)
    {
        currentCamera->setCurrentSide(sofa::component::visualmodel::BaseCamera::LEFT);
        DisplayOBJs();
    }
    if(supportStereo)
    {
        if (stereo)
        {
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
        }
    }
    //2nd pass
    if (twopass)
    {
        if (viewport)
        {
            sofa::core::visual::VisualParams::Viewport vp = vpright;
            vparams->viewport() = vp;
            glViewport(vp[0], vp[1], vp[2], vp[3]);
            glScissor(vp[0], vp[1], vp[2], vp[3]);
            glEnable(GL_SCISSOR_TEST);
        }
        if (stencil)
        {
            glStencilFunc(GL_NOTEQUAL, 0x1, 0x1);
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        }

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        if(sStrat == sofa::component::visualmodel::BaseCamera::PARALLEL) {glTranslated(-sShift/2,0,0);}

        glMultMatrixd(mat);
        if (_renderingMode == GL_RENDER)
        {
            currentCamera->setCurrentSide(sofa::component::visualmodel::BaseCamera::RIGHT);
            DisplayOBJs();
        }
        if (stereo)
        {
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
        }
        if (viewport)
        {
            vparams->viewport() = sofa::helper::make_array(0,0,_W,_H);
            glViewport(0, 0, _W, _H);
            glScissor(0, 0, _W, _H);
            glDisable(GL_SCISSOR_TEST);
        }
        if (stencil)
        {
            glDisable(GL_STENCIL_TEST);
        }
    }
    DisplayMenu(); // always needs to be the last object being drawn

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

    calcProjection(width, height);

    emit( resizeW(_W));
    emit( resizeH(_H));
}

// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void QtViewer::calcProjection(int width, int height)
{
    if (!width) width = _W;
    if (!height) height = _H;

    /// Camera part
    if (!currentCamera)
        return;

    if (groot && (!groot->f_bbox.getValue().isValid() || _axis))
    {
        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
    }
    currentCamera->computeZ();
    currentCamera->p_widthViewport.setValue(width);
    currentCamera->p_heightViewport.setValue(height);

    GLdouble projectionMatrix[16];
    currentCamera->getOpenGLProjectionMatrix(projectionMatrix);

    glViewport(0, 0, width * this->devicePixelRatio(), height * this->devicePixelRatio()); // to handle retina displays
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMultMatrixd(projectionMatrix);

    glMatrixMode(GL_MODELVIEW);
    glGetDoublev(GL_PROJECTION_MATRIX, lastProjectionMatrix);

    //Update vparams
    vparams->zNear() = currentCamera->getZNear();
    vparams->zFar() = currentCamera->getZFar();
    vparams->viewport() = sofa::helper::make_array(0, 0, width, height);
    vparams->setProjectionMatrix(projectionMatrix);
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
    drawScene();

    if(!captureTimer.isActive())
        SofaViewer::captureEvent();

    if (_waitForRender)
        _waitForRender = false;

    emit( redrawn());
}

void QtViewer::paintEvent(QPaintEvent* qpe)
{
    QOpenGLWidget::paintEvent(qpe );
/*
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::HighQualityAntialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, false);
*/
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
    const sofa::defaulttype::BoundingBox sceneBBox = vparams->sceneBBox();
    if (sceneBBox.isValid() && ! sceneBBox.isFlat())
        coeffDeplacement *= 0.001 * (sceneBBox.maxBBox()
                - sceneBBox.minBBox()).norm();
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
    if (isControlPressed()) // pass event to the scene data structure
    {
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
        }
        update();
        }
}

void QtViewer::keyReleaseEvent(QKeyEvent * e)
{
    SofaViewer::keyReleaseEvent(e);
}

void QtViewer::wheelEvent(QWheelEvent* e)
{
    SofaViewer::wheelEvent(e);
}

void QtViewer::mousePressEvent(QMouseEvent * e)
{
    mouseEvent(e);

    SofaViewer::mousePressEvent(e);
}

void QtViewer::mouseReleaseEvent(QMouseEvent * e)
{
    mouseEvent(e);

    SofaViewer::mouseReleaseEvent(e);

}

void QtViewer::mouseMoveEvent(QMouseEvent * e)
{

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
    else if (e->modifiers() & Qt::ShiftModifier)
    {
        isInteractive = true;
        SofaViewer::mouseEvent(e);
    }
    else if (e->modifiers() & Qt::ControlModifier)
    {
        isInteractive = true;
    }
    else if (e->modifiers() & Qt::AltModifier)
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
                //g_DepthBias[1] += dy * 0.01;
                //std::cout << "Depth bias = " << g_DepthBias[0] << " "
                //          << g_DepthBias[1] << std::endl;
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
                //g_DepthOffset[0] += dx * 0.01;
                //g_DepthOffset[1] += dy * 0.01;
                //std::cout << "Depth offset = " << g_DepthOffset[0] << " "
                //          << g_DepthOffset[1] << std::endl;
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

    const sofa::core::visual::VisualParams::Viewport& viewport = vparams->viewport();

    Vec3d p0, px, py, pz, px1, py1;
    gluUnProject(eventX, viewport[3] - 1 - (eventY), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(p0[0]), &(p0[1]), &(p0[2]));
    gluUnProject(eventX + 1, viewport[3] - 1 - (eventY), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(px[0]), &(px[1]), &(px[2]));
    gluUnProject(eventX, viewport[3] - 1 - (eventY + 1), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(py[0]), &(py[1]), &(py[2]));
    gluUnProject(eventX, viewport[3] - 1 - (eventY), 0.1,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(pz[0]), &(pz[1]), &(pz[2]));
    gluUnProject(eventX + 1, viewport[3] - 1 - (eventY), 0.1,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(px1[0]), &(px1[1]), &(px1[2]));
    gluUnProject(eventX, viewport[3] - 1 - (eventY + 1), 0,
            lastModelviewMatrix, lastProjectionMatrix,
            viewport.data(), &(py1[0]), &(py1[1]), &(py1[2]));
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
    getPickHandler()->updateRay(position, direction);
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

void QtViewer::getView(Vector3& pos, Quat& ori) const
{
    SofaViewer::getView(pos, ori);
}

void QtViewer::setView(const Vector3& pos, const Quat &ori)
{
    SofaViewer::setView(pos, ori);
}

void QtViewer::moveView(const Vector3& pos, const Quat &ori)
{
    SofaViewer::moveView(pos, ori);
}

void QtViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + "." + VIEW_FILE_EXTENSION;
        if(currentCamera->exportParametersInFile(viewFileName))
            std::cout << "View parameters saved in " << viewFileName << std::endl;
        else
            std::cout << "Error while saving view parameters in " << viewFileName << std::endl;
    }
}

void QtViewer::setSizeW(int size)
{
    resizeGL(size, _H);
    update();
}

void QtViewer::setSizeH(int size)
{
    resizeGL(_W, size);
    update();
}

//void QtViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
//{
//    SofaViewer::setCameraMode(mode);

//    switch (mode)
//    {
//    case core::visual::VisualParams::ORTHOGRAPHIC_TYPE:
//        camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
//        break;
//    case core::visual::VisualParams::PERSPECTIVE_TYPE:
//        camera()->setType( qglviewer::Camera::PERSPECTIVE  );
//        break;
//    }
//}

QString QtViewer::helpString() const
{
    static QString
    text(
        "<H1>QtViewer</H1><hr>\
<ul>\
<li><b>Mouse</b>: TO NAVIGATE<br></li>\
<li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
<li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
<li><b>C</b>: TO SWITCH INTERACTION MODE: press the KEY C.<br>\
Allow or not the navigation with the mouse.<br></li>\
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
