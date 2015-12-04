/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "EditorViewer.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/gui/ColourPickingVisitor.h>
//#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>
#include <sofa/gui/BaseGUI.h>
#include <qevent.h>
#include <sofa/gui/qt/GenGraphForm.h>

#include <sofa/helper/gl/glfont.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/io/ImageBMP.h>

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

namespace qgl
{

using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using sofa::simulation::getSimulation;
using namespace sofa::simulation;

helper::SofaViewerCreator<EditorViewer> EditorViewer_class("editor",false);
SOFA_DECL_CLASS ( EditorViewerGUI )


//static bool LeftPressedForMove = false;
//static bool RightPressedForMove = false;


// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------

QGLFormat EditorViewer::setupGLFormat(const unsigned int nbMSAASamples)
{
    QGLFormat f = QGLFormat::defaultFormat();

    if(nbMSAASamples > 1)
    {
        f.setSampleBuffers(true);
        f.setSamples(nbMSAASamples);
    }

    return f;
}

EditorViewer::EditorViewer(QWidget* parent, const char* name, const unsigned int nbMSAASamples)
    : QGLViewer(setupGLFormat(nbMSAASamples), parent)
{
    this->setObjectName(name);

    groot = NULL;
    initTexturesDone = false;

    backgroundColour[0]=1.0f;
    backgroundColour[1]=1.0f;
    backgroundColour[2]=1.0f;

    // setup OpenGL mode for the window
    //Fl_Gl_Window::mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA);
    timerAnimate = new QTimer(this);
    connect( timerAnimate, SIGNAL(timeout()), this, SLOT(animate()) );

    //	_previousEyePos = Vector3(0.0, 0.0, 0.0);
    // 	_zoom = 1.0;
    // 	_zoomSpeed = 250.0;
    // 	_panSpeed = 25.0;
    _video = false;
    _axis = false;
    _background = 0;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;

    _waitForRender=false;

    //////////////////////

    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
    m_isControlPressed = false;

    setManipulatedFrame( new qglviewer::ManipulatedFrame() );
    //near and far plane are better placed
    camera()->setZNearCoefficient(0.001);
    camera()->setZClippingCoefficient(5);

    vparams->zNear() = camera()->zNear();
    vparams->zFar()  = camera()->zFar();


    ///////////////////////
    m_selecting = false;
    m_dragging = false;
    m_draginit = false;
    m_selectionMethod = SingleCell;
    m_navigating = false;


    sofa::core::objectmodel::BaseObjectDescription descriptionSphereROI("mSphereROI","SphereROI");
    sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(getSimulation()->GetRoot()->getContext(), &descriptionSphereROI);
    m_sphereSelection = dynamic_cast<sofa::component::engine::SphereROI<defaulttype::Vec3dTypes>* >(obj.get());

    connect( &captureTimer, SIGNAL(timeout()), this, SLOT(captureEvent()) );
}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
EditorViewer::~EditorViewer()
{
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void EditorViewer::init(void)
{
    restoreStateFromFile();


    static	 GLfloat	specref[4];
    static	 GLfloat	ambientLight[4];
    static	 GLfloat	diffuseLight[4];
    static	 GLfloat	specular[4];
    static	 GLfloat	lmodel_ambient[]	= {0.0f, 0.0f, 0.0f, 0.0f};
    static	 GLfloat	lmodel_twoside[]	= {GL_FALSE};
    static	 GLfloat	lmodel_local[]		= {GL_FALSE};
    bool		initialized			= false;

    if (!initialized)
    {
        //std::cout << "progname="<<sofa::gui::qt::progname<<std::endl;
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

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

        _beginTime = helper::system::thread::CTime::getTime();

        printf("\n");


        // GL_LIGHT1 follows the camera
        // 	glMatrixMode(GL_MODELVIEW);
        // 	glPushMatrix();
        // 	glLoadIdentity();
        // 	glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
        // 	glPopMatrix();



        // 	camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
        // 	camera()->setType( qglviewer::Camera::PERSPECTIVE  );

    }

    // switch to preset view

    resetView();

    // Redefine keyboard events
    // The default SAVE_SCREENSHOT shortcut is Ctrl+S and this shortcut is used to
    // save x3d file in the MainController. So we need to change it:
    setShortcut(QGLViewer::SAVE_SCREENSHOT, Qt::Key_S);
    setShortcut(QGLViewer::HELP, Qt::Key_H);
    // Disable ESC shortcut
    setShortcut(QGLViewer::EXIT_VIEWER, 0);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void EditorViewer::PrintString(void* font, char* string)
{
    int	len, i;

    len = (int) strlen(string);
    for (i = 0; i < len; i++)
    {
        glutBitmapCharacter(font, string[i]);
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void EditorViewer::Display3DText(float x, float y, float z, char* string)
{
    char*	c;

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
void EditorViewer::DrawAxis(double xpos, double ypos, double zpos,
        double arrowSize)
{
    glPushMatrix();
    glTranslatef(xpos, ypos,zpos);
    QGLViewer::drawAxis(arrowSize);
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void EditorViewer::DrawBox(Real* minBBox, Real* maxBBox, Real r)
{
    //std::cout << "box = < " << minBBox[0] << ' ' << minBBox[1] << ' ' << minBBox[2] << " >-< " << maxBBox[0] << ' ' << maxBBox[1] << ' ' << maxBBox[2] << " >"<< std::endl;
    if (r==0.0)
        r = (Vector3(maxBBox) - Vector3(minBBox)).norm() / 500;
#if 0
    {
        Enable<GL_DEPTH_TEST> depth;
        Disable<GL_LIGHTING> lighting;
        glColor3f(0.0, 1.0, 1.0);
        glBegin(GL_LINES);
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d(           minBBox[0]           ,
                    (corner&1)?minBBox[1]:maxBBox[1],
                    (corner&2)?minBBox[2]:maxBBox[2]);
            glVertex3d(           maxBBox[0]           ,
                    (corner&1)?minBBox[1]:maxBBox[1],
                    (corner&2)?minBBox[2]:maxBBox[2]);
        }
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    minBBox[1]           ,
                    (corner&2)?minBBox[2]:maxBBox[2]);
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    maxBBox[1]           ,
                    (corner&2)?minBBox[2]:maxBBox[2]);
        }

        // --- Draw the Z edges
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    minBBox[2]           );
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    maxBBox[2]           );
        }
        glEnd();
        return;
    }
#endif
    Enable<GL_DEPTH_TEST> depth;
    Enable<GL_LIGHTING> lighting;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the corners
    glColor3f(0.0, 1.0, 1.0);
    for (int corner=0; corner<8; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                (corner&4)?minBBox[2]:maxBBox[2]);
        gluSphere(_sphere,2*r,20,10);
        glPopMatrix();
    }

    glColor3f(1.0, 1.0, 0.0);
    // --- Draw the X edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated(           minBBox[0]           ,
                (corner&1)?minBBox[1]:maxBBox[1],
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(90,0,1,0);
        gluCylinder(_tube, r, r, maxBBox[0] - minBBox[0], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Y edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                minBBox[1]           ,
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(-90,1,0,0);
        gluCylinder(_tube, r, r, maxBBox[1] - minBBox[1], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Z edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                minBBox[2]           );
        gluCylinder(_tube, r, r, maxBBox[2] - minBBox[2], 10, 10);
        glPopMatrix();
    }
}


// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void EditorViewer::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
        double ymax, double step)
{
    /*register*/ double x, y;

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
void EditorViewer::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
        double zmax, double step)
{
    /*register*/ double y, z;
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
void EditorViewer::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
        double zmax, double step)
{
    /*register*/ double x, z;
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

void EditorViewer::drawColourPicking(ColourPickingVisitor::ColourCode code)
{

    // Define background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // GL_PROJECTION matrix
    camera()->loadProjectionMatrix();
    // GL_MODELVIEW matrix
    camera()->loadModelViewMatrix();



    ColourPickingVisitor cpv(sofa::core::visual::VisualParams::defaultInstance(), code);
    cpv.execute(groot.get());

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void EditorViewer::DrawLogo()
{
    glPushMatrix();

    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage())
    {
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();
//        h = _H;
//        w = _W;
    }
    else return;

    Enable <GL_TEXTURE_2D> tex;
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
    glVertex3d((_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 0.0);
    glVertex3d( _W-(_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 1.0);
    glVertex3d( _W-(_W-w)/2, _H-(_H-h)/2, 0.0);

    glTexCoord2d(0.0, 1.0);
    glVertex3d((_W-w)/2, _H-(_H-h)/2, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void EditorViewer::DisplayOBJs()
{


    if (_background==0)
        DrawLogo();

    if (!groot) return;

    // 		// Initialize lighting
    glPushMatrix();
    glLoadIdentity();
    glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
    glPopMatrix();
    Enable<GL_LIGHT0> light0;
    //
    glColor3f(0.5f, 0.5f, 0.6f);
    // 			DrawXZPlane(-4.0, -20.0, 20.0, -20.0, 20.0, 1.0);
    // 			DrawAxis(0.0, 0.0, 0.0, 10.0);


    if (!groot->f_bbox.getValue().isValid()) viewAll();


    sofa::defaulttype::BoundingBox& bbox = vparams->sceneBBox();
    bbox = groot->f_bbox.getValue();

    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1,1,1,1);
    glDisable(GL_COLOR_MATERIAL);

    if (!initTexturesDone)
    {
        //		std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        simulation::getSimulation()->initTextures(groot.get());
        //---------------------------------------------------
        initTexturesDone = true;
    }


    {
        //Draw Debug information of the components
        simulation::getSimulation()->draw(vparams,groot.get());
        if (_axis)
        {
            this->setSceneBoundingBox(qglviewer::Vec(vparams->sceneBBox().minBBoxPtr()),
                    qglviewer::Vec(vparams->sceneBBox().maxBBoxPtr()) );

            //DrawAxis(0.0, 0.0, 0.0, 10.0);
            DrawAxis(0.0, 0.0, 0.0, this->sceneRadius());

            if (vparams->sceneBBox().isValid())
                DrawBox(vparams->sceneBBox().minBBoxPtr(), vparams->sceneBBox().maxBBoxPtr());
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void EditorViewer::DisplayMenu(void)
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

void EditorViewer::MakeStencilMask()
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
void EditorViewer::drawScene(void)
{

    camera()->getProjectionMatrix( lastProjectionMatrix );
    sofa::core::visual::VisualParams::Viewport& viewport = vparams->viewport();
    viewport[0] = 0;
    viewport[1] = 0;
    viewport[2] = camera()->screenWidth();
    viewport[3] = camera()->screenHeight();

    vparams->zFar() = camera()->zFar();
    vparams->zNear() = camera()->zNear();

    camera()->getModelViewMatrix( lastModelviewMatrix );
    vparams->setModelViewMatrix( lastModelviewMatrix );
    vparams->setProjectionMatrix( lastProjectionMatrix );
    //camera()->frame()->getMatrix( lastModelviewMatrix );

    //for(int i=0 ; i<16 ;i++)
    //	std::cout << lastModelviewMatrix[i] << " ";
    //
    //std::cout << std::endl;
    //std::cout << "P " << camera()->position().x << " " << camera()->position().y << " " << camera()->position().z << " " << std::endl;
    //std::cout << "T " << camera()->frame()->translation().x << " " << camera()->frame()->translation().y << " " << camera()->frame()->translation().z << " " << std::endl;
    //std::cout << "Q " << camera()->orientation() << std::endl;
    //std::cout << "R " << camera()->frame()->rotation() << " " << std::endl;

    if (_renderingMode == GL_RENDER)
    {

        if(m_selecting)
            m_sphereSelection->draw(vparams);
        DisplayOBJs();
        DisplayMenu();		// always needs to be the last object being drawn
    }

}

void EditorViewer::viewAll()
{
    if (!groot) return;
    sofa::defaulttype::BoundingBox& bbox = vparams->sceneBBox();
    bbox = groot->f_bbox.getValue();


    if (bbox.minBBox().x() == bbox.maxBBox().x() || !bbox.isValid())
    {
        bbox.minBBox().x() = -1;
        bbox.maxBBox().x() =  1;
    }
    if (bbox.minBBox().y() == bbox.maxBBox().y() || !bbox.isValid())
    {
        bbox.minBBox().y() = -1;
        bbox.maxBBox().y() =  1;
    }
    if (bbox.minBBox().z() == bbox.maxBBox().z() || !bbox.isValid())
    {
        bbox.minBBox().z() = -1;
        bbox.maxBBox().z() =  1;
    }

    QGLViewer::setSceneBoundingBox(   qglviewer::Vec(bbox.minBBoxPtr()),qglviewer::Vec(bbox.maxBBoxPtr())) ;

    qglviewer::Vec pos;
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 75.0;
    camera()->setPosition(pos);
    camera()->showEntireScene();
}



// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void EditorViewer::resizeGL(int width, int height)
{
    _W = width;
    _H = height;


    QGLViewer::resizeGL( width,  height);
    // 	    camera()->setScreenWidthAndHeight(_W,_H);

    this->resize(width, height);
    emit( resizeW( _W ) );
    emit( resizeH( _H ) );
}


// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void EditorViewer::draw()
{
    //	ctime_t beginDisplay;
    //ctime_t endOfDisplay;

    //	beginDisplay = MesureTemps();

    // valid() is turned off when FLTK creates a new context for this window
    // or when the window resizes, and is turned on after draw() is called.
    // Use this to avoid unneccessarily initializing the OpenGL context.
    //static double lastOrthoTransZ = 0.0;
    /*
    if (!valid())
    {
    InitGFX();		// this has to be called here since we don't know when the context is created
    _W = w();
    _H = h();
    reshape(_W, _H);
    }
    */
    // clear buffers (color and depth)
    if (_background==0)
        glClearColor(0.0f,0.0f,0.0f,1.0f);
    else if (_background==1)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    else if (_background==2)
        glClearColor(backgroundColour[0],backgroundColour[1],backgroundColour[2], 1.0f);
    glClearDepth(1.0);
    glClear(_clearBuffer);

    // draw the scene
    drawScene();

    //draw the sphereROI if in selection mode and sphere selectioned

    if(!captureTimer.isActive())
        SofaViewer::captureEvent();

    if (_waitForRender)
        _waitForRender = false;

    emit( redrawn() );
}

void EditorViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    SofaViewer::setCameraMode(mode);

    switch (mode)
    {
    case core::visual::VisualParams::ORTHOGRAPHIC_TYPE:
        camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
        break;
    case core::visual::VisualParams::PERSPECTIVE_TYPE:
        camera()->setType( qglviewer::Camera::PERSPECTIVE  );
        break;
    }
}


// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------


void EditorViewer::keyPressEvent ( QKeyEvent * e )
{

    //Tracking Mode

//    std::cerr<<"QtGLViewer::keyPressEvent, get "<<e->key()<<std::endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
//        std::cerr<<"QtGLViewer::keyPressEvent, key = "<<e->key()<<" with Control pressed "<<std::endl;
        if (groot)
        {
            sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
            groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        }
    }
    else  // control the GUI
    {
        //                            cerr<<"QtGLViewer::keyPressEvent, key = "<<e->key()<<" without Control pressed "<<endl;
        switch(e->key())
        {
        case Qt::Key_A: //axis
        case Qt::Key_S: //sofa screenshot
        case Qt::Key_H: //shortcuts for screenshot and help page specified for qglviewer
        {
            QGLViewer::keyPressEvent(e);
            break;
        }
        case Qt::Key_C:
        {
            viewAll();
            break;
        }
        case Qt::Key_Z:
        {
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT,viewport);
            getPickHandler()->activateRay(viewport[2],viewport[3], groot.get());

            helper::vector<defaulttype::Vec<3,Real> > center;
            center.push_back(getPickHandler()->getLastPicked()->point);
            std::cout << getPickHandler()->getLastPicked()->point << std::endl;

//            m_sphereSelection->centers.setValue(center);
            setMouseTracking(true);
            m_selecting = true;
            break;
        }
        case Qt::Key_R:
        {
            m_navigating = true;
            break;
        }
        case Qt::Key_D:
        {
            if(!m_dragging)
            {
                //getSelected cells

                m_dragging = true;
                m_draginit = false;
                setMouseTracking(true);
            }
            else
            {
                m_dragging = false;
                m_draginit = false;
                setMouseTracking(false);
            }
            break;
        }
        default:
        {
            SofaViewer::keyPressEvent(e);
            QGLViewer::keyPressEvent(e);
        }
        }
    }
    update();
}

void EditorViewer::keyReleaseEvent ( QKeyEvent * e )
{

    switch(e->key())
    {
        case Qt::Key_Z:
        {
            getPickHandler()->deactivateRay();
            setMouseTracking(false);
            m_selecting = false;
            break;
        }
        case Qt::Key_R:
        {
            m_navigating = false;
            break;
        }
        default:
        {
            QGLViewer::keyReleaseEvent(e);
            SofaViewer::keyReleaseEvent(e);
        }

    }
}

void EditorViewer::mousePressEvent ( QMouseEvent * e )
{
    //1. selection
    if(m_selecting && (e->button() == Qt::LeftButton || e->button() == Qt::RightButton))
    {
        //only vertices for now

        switch(m_selectionMethod)
        {
            case SingleCell:
            {
                if(e->button() == Qt::LeftButton)
                {
                    std::cout << "select the vertex" << std::endl;

                    //code for the ray selection

                    GLint viewport[4];
                    glGetIntegerv(GL_VIEWPORT,viewport);

                    MousePosition mousepos;
                    mousepos.screenWidth  = viewport[2];
                    mousepos.screenHeight = viewport[3];
                    mousepos.x      = e->x();
                    mousepos.y      = e->y();

                    getPickHandler()->activateRay(viewport[2],viewport[3], groot.get());
                    getPickHandler()->updateMouse2D( mousepos );


                    switch (e->type())
                    {
                    case QEvent::MouseButtonPress:
                        if (e->button() == Qt::LeftButton)
                        {
                            getPickHandler()->handleMouseEvent(PRESSED, LEFT);
                        }
                        else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
                        {
                            getPickHandler()->handleMouseEvent(PRESSED, RIGHT);
                        }
                        else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
                        {
                            getPickHandler()->handleMouseEvent(PRESSED, MIDDLE);
                        }
                        break;
                    case QEvent::MouseButtonRelease:
                        //if (e->button() == Qt::LeftButton)
                    {

                        if (e->button() == Qt::LeftButton)
                        {
                            getPickHandler()->handleMouseEvent(RELEASED, LEFT);
                        }
                        else if (e->button() == Qt::RightButton)
                        {
                            getPickHandler()->handleMouseEvent(RELEASED, RIGHT);
                        }
                        else if (e->button() == Qt::MidButton)
                        {
                            getPickHandler()->handleMouseEvent(RELEASED, MIDDLE);
                        }
                    }
                    break;
                    default:
                        break;
                    }
                    moveRayPickInteractor(e->x(), e->y());
                }
                else if(e->button() == Qt::RightButton)
                    std::cout << "deselect the vertex" << std::endl;

                break;
            }
            case WithinSphere:
            {
                std::cout << "select within sphere" << std::endl;
                break;
            }
            default: // to fix a warning
                break;
        }
    }
    else
//    if( ! mouseEvent(e) )
        SofaViewer::mousePressEvent(e);
        QGLViewer::mousePressEvent(e);
}

void EditorViewer::mouseReleaseEvent ( QMouseEvent * e )
{
//    if( ! mouseEvent(e) )
        SofaViewer::mouseReleaseEvent(e);
        QGLViewer::mouseReleaseEvent(e);
}

void EditorViewer::mouseMoveEvent ( QMouseEvent * e )
{
    //1. selection

    if(m_dragging)
    {
        std::cout << "draging" << std::endl;
        //_sceneTransform.ApplyInverse();
        switch (e->type())
        {
        case QEvent::MouseButtonPress:
            if (e->button() == Qt::LeftButton)
            {
                getPickHandler()->handleMouseEvent(PRESSED, LEFT);
            }
            else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
            {
                getPickHandler()->handleMouseEvent(PRESSED, RIGHT);
            }
            else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
            {
                getPickHandler()->handleMouseEvent(PRESSED, MIDDLE);
            }
            break;
        case QEvent::MouseButtonRelease:
            //if (e->button() == Qt::LeftButton)
        {

            if (e->button() == Qt::LeftButton)
            {
                getPickHandler()->handleMouseEvent(RELEASED, LEFT);
            }
            else if (e->button() == Qt::RightButton)
            {
                getPickHandler()->handleMouseEvent(RELEASED, RIGHT);
            }
            else if (e->button() == Qt::MidButton)
            {
                getPickHandler()->handleMouseEvent(RELEASED, MIDDLE);
            }
        }
        break;
        default:
            break;
        }
        moveRayPickInteractor(e->x(), e->y());

    }
    //else
    //    if( ! mouseEvent(e) )
            SofaViewer::mouseMoveEvent(e);
            QGLViewer::mouseMoveEvent(e);
}

bool EditorViewer::mouseEvent(QMouseEvent * e)
{
    if(e->modifiers()&Qt::ShiftModifier)
    {
        SofaViewer::mouseEvent(e);
        return true;
    }

    return false;
}

void EditorViewer::wheelEvent(QWheelEvent* e)
{
    if(m_selecting)
    {
        switch(m_selectionMethod)
        {
            case SingleCell:
                break;
            case WithinSphere:
            {
                if(e->delta() > 0)
                    selection_radiusCoef *= 0.9f;
                else
                    selection_radiusCoef *= 1.1f;

                break;
            }
            default: // to fix a warning
                break;
        }
    }
    else if(m_navigating)
    {
        sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Wheel, e->delta());
        groot->propagateEvent(core::ExecParams::defaultInstance(), &me);
    }
    else if(e->modifiers() & Qt::ShiftModifier)
    {
        std::cout << "editing mode" << std::endl;

        sofa::core::topology::BaseMeshTopology* topo;
        topo = groot->getMeshTopology();
    }
    else
        QGLViewer::wheelEvent(e);
}

void EditorViewer::moveRayPickInteractor(int eventX, int eventY)
{
    const sofa::core::visual::VisualParams::Viewport& viewport = vparams->viewport();
    Vec3d p0, px, py, pz, px1, py1;
    gluUnProject(eventX,   viewport[3]-1-(eventY),   0,   lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(p0[0]),  &(p0[1]),  &(p0[2]));
    gluUnProject(eventX+1, viewport[3]-1-(eventY),   0,   lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(px[0]),  &(px[1]),  &(px[2]));
    gluUnProject(eventX,   viewport[3]-1-(eventY+1), 0,   lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(py[0]),  &(py[1]),  &(py[2]));
    gluUnProject(eventX,   viewport[3]-1-(eventY),   0.1, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(pz[0]),  &(pz[1]),  &(pz[2]));
    gluUnProject(eventX+1, viewport[3]-1-(eventY),   0.1, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(px1[0]), &(px1[1]), &(px1[2]));
    gluUnProject(eventX,   viewport[3]-1-(eventY+1), 0,   lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(py1[0]), &(py1[1]), &(py1[2]));
    px1 -= pz;
    py1 -= pz;
    px -= p0;
    py -= p0;
    pz -= p0;
    double r0 = sqrt(px.norm2() + py.norm2());
    double r1 = sqrt(px1.norm2() + py1.norm2());
    r1 = r0 + (r1-r0) / pz.norm();
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
    Mat3x3d mat; mat = transform;
    Quat q; q.fromMatrix(mat);


    Vec3d position, direction;
    position  = transform*Vec4d(0,0,0,1);
    direction = transform*Vec4d(0,0,1,0);
    direction.normalize();
    pick->updateRay(position, direction);
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void EditorViewer::resetView()
{
    viewAll();

    if (!sceneFileName.empty())
    {
        //Test if we have a specific view point for the QGLViewer
        //That case, the camera will be well placed
        std::string viewFileName = sceneFileName+"."+sofa::gui::BaseGUI::GetGUIName()+".view";
        std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
            qglviewer::Vec pos;
            in >> pos[0];
            in >> pos[1];
            in >> pos[2];

            qglviewer::Quaternion q;
            in >> q[0];
            in >> q[1];
            in >> q[2];
            in >> q[3];
            q.normalize();

            camera()->setOrientation(q);
            camera()->setPosition(pos);

            in.close();
            update();

            return;
        }
        else
        {
            //If we have the default QtViewer view file, we have to use, showEntireScene
            //as the FOV of the QtViewer is not constant, so the parameters are not good
            std::string viewFileName = sceneFileName+".view";
            std::ifstream in(viewFileName.c_str());
            if (!in.fail())
            {
                qglviewer::Vec pos;
                in >> pos[0];
                in >> pos[1];
                in >> pos[2];

                qglviewer::Quaternion q;
                in >> q[0];
                in >> q[1];
                in >> q[2];
                in >> q[3];
                q.normalize();

                camera()->setOrientation(q);
                camera()->setPosition(pos);
                camera()->showEntireScene();

                in.close();
                update();

                return;
            }
        }
    }
    update();
}

void EditorViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+"."+sofa::gui::BaseGUI::GetGUIName()+".view";
        std::ofstream out(viewFileName.c_str());
        if (!out.fail())
        {
            out << camera()->position()[0] << " " << camera()->position()[1] << " " << camera()->position()[2] << "\n";
            out << camera()->orientation()[0] << " " << camera()->orientation()[1] << " " << camera()->orientation()[2] << " " << camera()->orientation()[3] << "\n";
            out.close();
        }
        std::cout << "View parameters saved in "<<viewFileName<<std::endl;
    }
}

void EditorViewer::getView(Vec3d& pos, Quat& ori) const
{
    qglviewer::Vec position = camera()->position();
    for(int i = 0; i < 3; ++i) pos[i] = position[i];
    qglviewer::Quaternion orientation = camera()->orientation();
    for(int i = 0; i < 4; ++i) ori[i] = orientation[i];
}

void EditorViewer::setView(const Vec3d& pos, const Quat &ori)
{
    camera()->setPosition(qglviewer::Vec(pos[0],pos[1],pos[2]));
    camera()->setOrientation(qglviewer::Quaternion(ori[0],ori[1],ori[2],ori[3]));
}

void EditorViewer::setSizeW( int size )
{
    resizeGL( size, _H );
    updateGL();
}

void EditorViewer::setSizeH( int size )
{
    resizeGL( _W, size );
    updateGL();

}

QString EditorViewer::helpString() const
{

    static QString text(
        (QString)"<H1>EditorViewer</H1><hr>\
                <ul>\
                <li><b>Mouse</b>: TO NAVIGATE<br></li>\
                <li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
                <li><b>A</b>: TO DRAW AXIS<br></li>\
                <li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
                <li><b>C</b>: TO CENTER THE VIEW<br></li>\
                <li><b>H</b>: TO OPEN HELP of QGLViewer<br></li>\
                <li><b>O</b>: TO EXPORT TO .OBJ<br>\
                The generated files scene-time.obj and scene-time.mtl are saved in the running project directory<br></li>\
                <li><b>P</b>: TO SAVE A SEQUENCE OF OBJ<br>\
                Each time the frame is updated an obj is exported<br></li>\
                <li><b>R</b>: TO DRAW THE SCENE AXIS<br></li>\
                <li><b>T</b>: TO CHANGE BETWEEN A PERSPECTIVE OR AN ORTHOGRAPHIC CAMERA<br></li>\
                The captured images are saved in the running project directory under the name format capturexxxx.bmp<br></li>\
                <li><b>S</b>: TO SAVE A SCREENSHOT<br>\
                <li><b>V</b>: TO SAVE A VIDEO<br>\
                Each time the frame is updated a screenshot is saved<br></li>\
                <li><b>Esc</b>: TO QUIT ::sofa:: <br></li></ul>"
#ifdef SOFA_HAVE_SENSABLE
        +(QString)"<H1>Sensable</H1>\
                <ul>\
                <li><b>Ctrl + index interface</b>: TO DRAW AXIS<br></li>\
                <li><b>Option OmniVisu</b>: TO DRAW INTERFACE<br></li>\
                <li>If <b>OmniVisu</b> and <b>axis</b> are active:<br></li>\
                <ul>\
                <li><b>Ctrl + Axis: x, y or z + arrow left or right</b>: TO MOVE THE INTERFACE ON THE AXIS<br></li>\
                <li><b>Ctrl + Axis: x, y or z + arrow up or down</b>: TO ROTATE THE INTERFACE ON THE AXIS<br></li>\
                <li><b>Ctrl + Q + arrow left or right</b>: TO CHANGE THE SCALE OF THE INTERFACE<br></li>\
                <li><b>Ctrl + E </b>: TO RESET THE POSITION OF THE INTERFACE<br></li>\
                </ul>\
                </ul>"
#endif
    );

    return text;
}





} // namespace qgl

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa
