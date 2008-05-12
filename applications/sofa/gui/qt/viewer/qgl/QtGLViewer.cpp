/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include "viewer/qgl/QtGLViewer.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/simulation/tree/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#ifdef WIN32
#include <GL/glaux.h>
#endif
#include <sofa/helper/system/glut.h>
#include <qevent.h>
#include "GenGraphForm.h"
#include "Main.h"

#include <sofa/helper/gl/glfont.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/gl/GLshader.h>
#include <sofa/helper/io/ImageBMP.h>

#ifdef SOFA_DEV

#include <sofa/simulation/automatescheduler/Automate.h>
#include <sofa/simulation/automatescheduler/CPU.h>
#include <sofa/simulation/automatescheduler/Node.h>
#include <sofa/simulation/automatescheduler/Edge.h>
#include <sofa/simulation/automatescheduler/ExecBus.h>
#include <sofa/simulation/automatescheduler/ThreadSimulation.h>

#endif // SOFA_DEV

#include <sofa/defaulttype/RigidTypes.h>


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

namespace qgl
{

using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using sofa::simulation::tree::getSimulation;
//extern UserInterface*	GUI;
//extern OBJmodel*		cubeModel;
// Quaternion QtGLViewer::_newQuat;


// Shadow Mapping parameters

// These store our width and height for the shadow texture
enum { SHADOW_WIDTH = 512 };
enum { SHADOW_HEIGHT = 512 };
enum { SHADOW_MASK_SIZE = 2048 };

// This is used to set the mode with glTexParameteri() for comparing depth values.
// We use GL_COMPARE_R_TO_TEXTURE_ARB as our mode.  R is used to represent the depth value.
#define GL_TEXTURE_COMPARE_MODE_ARB       0x884C

// This is used to set the function with glTexParameteri() to tell OpenGL how we
// will compare the depth values (we use GL_LEQUAL (less than or equal)).
#define GL_TEXTURE_COMPARE_FUNC_ARB       0x884D

// This mode is what will compare our depth values for shadow mapping
#define GL_COMPARE_R_TO_TEXTURE_ARB       0x884E

// The texture array where we store our image data
GLuint g_DepthTexture;

// This is our global shader object that will load the shader files
CShader g_Shader;

//float g_DepthOffset[2] = { 3.0f, 0.0f };
float g_DepthOffset[2] = { 10.0f, 0.0f };
float g_DepthBias[2] = { 0.0f, 0.0f };

// These are the light's matrices that need to be stored
float g_mProjection[16] = {0};
float g_mModelView[16] = {0};
//float g_mCameraInverse[16] = {0};

GLuint ShadowTextureMask;

// End of Shadow Mapping Parameters

static bool enabled = false;
sofa::core::ObjectFactory::ClassEntry* classVisualModel;

/// Activate this class of viewer.
/// This method is called before the viewer is actually created
/// and can be used to register classes associated with in the the ObjectFactory.
int QtGLViewer::EnableViewer()
{
    if (!enabled)
    {
        enabled = true;
        // Replace generic visual models with OglModel
        sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);
    }
    return 0;
}

/// Disable this class of viewer.
/// This method is called after the viewer is destroyed
/// and can be used to unregister classes associated with in the the ObjectFactory.
int QtGLViewer::DisableViewer()
{
    if (enabled)
    {
        enabled = false;
        sofa::core::ObjectFactory::ResetAlias("VisualModel", classVisualModel);
    }
    return 0;
}

// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
QtGLViewer::QtGLViewer(QWidget* parent, const char* name)
    : QGLViewer(parent, name)
{

    groot = NULL;
    initTexturesDone = false;
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
    _shadow = false;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;

    sceneBBoxIsValid = false;
    texLogo = NULL;

#ifdef SOFA_DEV

    _automateDisplayed = false;

#endif // SOFA_DEV

    /*_surfaceModel = NULL;
      _springMassView = NULL;
      _mapView = NULL;
      sphViewer = NULL;
    */
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



    //////////////////////


    interactor = NULL;
    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
    m_isControlPressed = false;

    setManipulatedFrame( new qglviewer::ManipulatedFrame() );
    //near and far plane are better placed
    camera()->setZNearCoefficient(0.001);
    camera()->setZClippingCoefficient(5);
}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
QtGLViewer::~QtGLViewer()
{
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void QtGLViewer::init(void)
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
        if (!GLEW_ARB_multitexture)
            std::cerr << "Error: GL_ARB_multitexture not supported\n";
#else
        glActiveTextureARB        = (PFNGLACTIVETEXTUREARBPROC)        glewGetProcAddress("glActiveTextureARB");
        glMultiTexCoord2fARB    = (PFNGLMULTITEXCOORD2FARBPROC)        glewGetProcAddress("glMultiTexCoord2fARB");

        // Make sure our multi-texturing extensions were loaded correctly
        if(!glActiveTextureARB || !glMultiTexCoord2fARB)
        {
            // Print an error message and quit.
            //    MessageBox(g_hWnd, "Your current setup does not support multitexturing", "Error", MB_OK);
            //PostQuitMessage(0);
        }
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
        texLogo = new helper::gl::Texture(new helper::io::ImageBMP( sofa::helper::system::DataRepository.getFile("textures/SOFA_logo.bmp") ));
        texLogo->init();

        glEnableClientState(GL_VERTEX_ARRAY);
        //glEnableClientState(GL_NORMAL_ARRAY);

        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

        // Here we allocate memory for our depth texture that will store our light's view
        CreateRenderTexture(g_DepthTexture, SHADOW_WIDTH, SHADOW_HEIGHT, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT);
        CreateRenderTexture(ShadowTextureMask, SHADOW_MASK_SIZE, SHADOW_MASK_SIZE, GL_LUMINANCE, GL_LUMINANCE);

        if (_shadow )
        {
            if ( !CShader::InitGLSL() ) { printf("WARNING QtGLViewer : shadows are not supported !\n"); _shadow=false;}
            else
            {
                // Here we pass in our new vertex and fragment shader files to our shader object.
                g_Shader.InitShaders(sofa::helper::system::DataRepository.getFile("shaders/ShadowMappingPCF.vert"), sofa::helper::system::DataRepository.getFile("shaders/ShadowMappingPCF.frag"));
            }
        }

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = CTime::getTime();

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
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------

///////////////////////////////// STORE LIGHT MATRICES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function positions our view from the light for shadow mapping
/////
///////////////////////////////// STORE LIGHT MATRICES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void QtGLViewer::StoreLightMatrices()
{
    //	_lightPosition[0] =  _sceneTransform.translation[0] + 10;//*cosf(TT);
    //	_lightPosition[1] =  _sceneTransform.translation[1] + 10;//*sinf(2*TT);
    //	_lightPosition[2] =  _sceneTransform.translation[2] + 35;//

    //_lightPosition[0] =  1;
    //_lightPosition[1] =  -10;
    //_lightPosition[2] =  0;

    // Reset our current light matrices
    memset(g_mModelView, 0, sizeof(float)*16);
    memset(g_mProjection, 0, sizeof(float)*16);

    g_mModelView[0] = 1; // identity
    g_mModelView[5] = 1;
    g_mModelView[10] = 1;
    g_mModelView[15] = 1;

    // Using perpective shadow map for the "miner lamp" case ( i.e. light.z == 0 )
    // which is just a "rotation" in sceen space

    float lx = -_lightPosition[0] * lastProjectionMatrix[0] - _lightPosition[1] * lastProjectionMatrix[4] + lastProjectionMatrix[12];
    float ly = -_lightPosition[0] * lastProjectionMatrix[1] - _lightPosition[1] * lastProjectionMatrix[5] + lastProjectionMatrix[13];
    float lz = -_lightPosition[0] * lastProjectionMatrix[2] - _lightPosition[1] * lastProjectionMatrix[6] + lastProjectionMatrix[14];
    //float lw = -_lightPosition[0] * lastProjectionMatrix[3] - _lightPosition[1] * lastProjectionMatrix[7] + lastProjectionMatrix[15];
    //std::cout << "lx = "<<lx<<" ly = "<<ly<<" lz = "<<lz<<" lw = "<<lw<<std::endl;

    Vector3 l(-lx,-ly,-lz);
    Vector3 y;
    y = l.cross(Vector3(1,0,0));
    Vector3 x;
    x = y.cross(l);
    l.normalize();
    y.normalize();
    x.normalize();

    g_mProjection[ 0] = x[0]; g_mProjection[ 4] = x[1]; g_mProjection[ 8] = x[2]; g_mProjection[12] =    0;
    g_mProjection[ 1] = y[0]; g_mProjection[ 5] = y[1]; g_mProjection[ 9] = y[2]; g_mProjection[13] =    0;
    g_mProjection[ 2] = l[0]; g_mProjection[ 6] = l[1]; g_mProjection[10] = l[2]; g_mProjection[14] =    0;
    g_mProjection[ 3] =    0; g_mProjection[ 7] =    0; g_mProjection[11] =    0; g_mProjection[15] =    1;

    g_mProjection[ 0] = x[0]; g_mProjection[ 4] = y[0]; g_mProjection[ 8] = l[0]; g_mProjection[12] =    0;
    g_mProjection[ 1] = x[1]; g_mProjection[ 5] = y[1]; g_mProjection[ 9] = l[1]; g_mProjection[13] =    0;
    g_mProjection[ 2] = x[2]; g_mProjection[ 6] = y[2]; g_mProjection[10] = l[2]; g_mProjection[14] =    0;
    g_mProjection[ 3] =    0; g_mProjection[ 7] =    0; g_mProjection[11] =    0; g_mProjection[15] =    1;

    glPushMatrix();
    {

        glLoadIdentity();
        glScaled(1.0/(fabs(g_mProjection[0])+fabs(g_mProjection[4])+fabs(g_mProjection[8])),
                1.0/(fabs(g_mProjection[1])+fabs(g_mProjection[5])+fabs(g_mProjection[9])),
                1.0/(fabs(g_mProjection[2])+fabs(g_mProjection[6])+fabs(g_mProjection[10])));
        glMultMatrixf(g_mProjection);
        glMultMatrixd(lastProjectionMatrix);

        // Grab the current matrix that will be used for the light's projection matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, g_mProjection);

        // Go back to the original matrix
    } glPopMatrix();

    /*
    // Let's push on a new matrix so we don't change the rest of the world
    glPushMatrix();{

    // Reset the current modelview matrix
    glLoadIdentity();

    // This is where we set the light's position and view.
    gluLookAt(_lightPosition[0],  _lightPosition[1],  _lightPosition[2],
    _sceneTransform.translation[0],	   _sceneTransform.translation[1],	    _sceneTransform.translation[2],		0, 1, 0);

    // Now that we have the light's view, let's save the current modelview matrix.
    glGetFloatv(GL_MODELVIEW_MATRIX, g_mModelView);

    // Reset the current matrix
    glLoadIdentity();

    // Set our FOV, aspect ratio, then near and far planes for the light's view
    gluPerspective(90.0f, 1.0f, 4.0f, 250.0f);

    // Grab the current matrix that will be used for the light's projection matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, g_mProjection);

    // Go back to the original matrix
    }glPopMatrix();
    */
}

/////////////////////////////// CREATE RENDER TEXTURE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function creates a blank texture to render to
/////
/////////////////////////////// CREATE RENDER TEXTURE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void QtGLViewer::CreateRenderTexture(GLuint& textureID, int sizeX, int sizeY, int channels, int type)
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Create the texture and store it on the video card
    glTexImage2D(GL_TEXTURE_2D, 0, channels, sizeX, sizeY, 0, type, GL_UNSIGNED_INT, NULL);

    // Set the texture quality
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

//////////////////////////////// APPLY SHADOW MAP \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function applies the shadow map to our world data
/////
//////////////////////////////// APPLY SHADOW MAP \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void QtGLViewer::ApplyShadowMap()
{
    // Let's turn our shaders on for doing shadow mapping on our world
    g_Shader.TurnOn();

    // Turn on our texture unit for shadow mapping and bind our depth texture
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_DepthTexture);

    // Give GLSL our texture unit that holds the shadow map
    g_Shader.SetInt(g_Shader.GetVariable("shadowMap"), 1);
    //g_Shader.SetInt(g_Shader.GetVariable("tex"), 0);

    // Here is where we set the mode and function for shadow mapping with shadow2DProj().

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB,
            GL_COMPARE_R_TO_TEXTURE_ARB);

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL);

    // Create our bias matrix to have a 0 to 1 ratio after clip space
    const float mBias[] = {0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.5+g_DepthBias[0], 0.0,
            0.5, 0.5, 0.5+g_DepthBias[1], 1.0
                          };

    glMatrixMode(GL_TEXTURE);

    glLoadMatrixf(mBias);			// The bias matrix to convert to a 0 to 1 ratio
    glMultMatrixf(g_mProjection);	// The light's projection matrix
    glMultMatrixf(g_mModelView);	// The light's modelview matrix
    //glMultMatrixf(g_mCameraInverse);// The inverse modelview matrix

    glMatrixMode(GL_MODELVIEW);			// Switch back to normal modelview mode

    glActiveTextureARB(GL_TEXTURE0_ARB);

    // Render the world that needs to be shadowed


    camera()->getModelViewMatrix( lastModelviewMatrix );

    DisplayOBJs();

    // Reset the texture matrix
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);

    // Turn the first multi-texture pass off

    glActiveTextureARB(GL_TEXTURE1_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE0_ARB);

    // Light expected, we need to turn our shader off since we are done
    g_Shader.TurnOff();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::PrintString(void* font, char* string)
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
void QtGLViewer::Display3DText(float x, float y, float z, char* string)
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
void QtGLViewer::DrawAxis(double xpos, double ypos, double zpos,
        double arrowSize)
{
    float	fontScale	= (float) (arrowSize / 600.0);

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
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 88),
            arrowSize / 15.0, arrowSize /
            5.0);
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
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 89),
            arrowSize / 15.0, arrowSize /
            5.0);
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
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 90),
            arrowSize / 15.0, arrowSize /
            5.0);
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
void QtGLViewer::DrawBox(Real* minBBox, Real* maxBBox, Real r)
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
void QtGLViewer::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
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
void QtGLViewer::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
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
void QtGLViewer::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
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
void QtGLViewer::DrawLogo()
{
    glPushMatrix();

    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage())
    {
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();
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
void QtGLViewer::DisplayOBJs(bool shadowPass)
{
    if (!groot) return;

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
        getSimulation()->initTextures(groot);
        //---------------------------------------------------
        initTexturesDone = true;
    }

#ifdef SOFA_DEV
    if (!groot->getMultiThreadSimulation())
#endif // SOFA_DEV
    {
        if (shadowPass)
            getSimulation()->drawShadows(groot);
        else
            getSimulation()->draw(groot);
        if (_axis)
        {
            DrawAxis(0.0, 0.0, 0.0, 10.0);
            if (sceneMinBBox[0] < sceneMaxBBox[0])
                DrawBox(sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        }
    }
#ifdef SOFA_DEV
    else
        automateDisplayVM();
#endif // SOFA_DEV

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void QtGLViewer::DisplayMenu(void)
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

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::DrawScene(void)
{

    camera()->getProjectionMatrix( lastProjectionMatrix );

    camera()->getViewport( lastViewport );
    lastViewport[1]=0;
    lastViewport[3]=-lastViewport[3];


#if 1
    if (_shadow)
    {
        //glGetDoublev(GL_MODELVIEW_MATRIX,lastModelviewMatrix);

        // Update the light matrices for it's current position
        StoreLightMatrices();

        // Set the current viewport to our texture size
        glViewport(0, 0, (int)SHADOW_WIDTH, (int)SHADOW_HEIGHT);

        // Clear the screen and depth buffer so we can render from the light's view
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Now we just need to set the matrices for the light before we render
        glMatrixMode(GL_PROJECTION);

        // Push on a matrix to make sure we can restore to the old matrix easily
        glPushMatrix();
        {
            // Set the current projection matrix to our light's projection matrix
            glLoadMatrixf(g_mProjection);

            // Load modelview mode to set our light's modelview matrix
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            {
                // Load the light's modelview matrix before we render to a texture
                glLoadMatrixf(g_mModelView);

                // Since we don't care about color when rendering the depth values to
                // the shadow-map texture, we disable color writing to increase speed.
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

                // This turns of the polygon offset functionality to fix artifacts.
                // Comment this out and run the program to see what artifacts I mean.
                glEnable(GL_POLYGON_OFFSET_FILL);
                glDisable(GL_BLEND);

                // Eliminate artifacts caused by shadow mapping
                //	glPolygonOffset(1.0f, 0.10f);
                glPolygonOffset(g_DepthOffset[0], g_DepthOffset[1]);

                // Render the world according to the light's view
                DisplayOBJs(true);

                // Now that the world is rendered, save the depth values to a texture
                glDisable(GL_BLEND);
                //glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, g_DepthTexture);

                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, (int)SHADOW_WIDTH, (int)SHADOW_HEIGHT);

                // We can turn color writing back on since we already stored the depth values
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

                // Turn off polygon offsetting
                glDisable(GL_POLYGON_OFFSET_FILL);

            } glPopMatrix();
            // Go back to the projection mode and restore the original matrix
            glMatrixMode(GL_PROJECTION);
            // Restore the original projection matrix
        } glPopMatrix();

        // Go back to modelview model to start drawing like normal
        glMatrixMode(GL_MODELVIEW);

        // Restore our normal viewport size to our screen width and height
        glViewport(0, 0, GetWidth(), GetHeight());

        // Clear the color and depth bits and start over from the camera's view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glPushMatrix();{
        //	glLoadIdentity();
        //	_sceneTransform.ApplyInverse();
        //	glGetFloatv(GL_MODELVIEW_MATRIX, g_mCameraInverse);
        //}glPopMatrix();

        glLightfv( GL_LIGHT0, GL_POSITION, _lightPosition );
        /*
          {
          glEnable(GL_TEXTURE_2D);
          glActiveTextureARB(GL_TEXTURE0_ARB);
          glBindTexture(GL_TEXTURE_2D, g_Texture[SHADOW_ID]);
          glTexEnvi(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,  GL_REPLACE);
          Disable<GL_DEPTH_TEST> dtoff;
          Disable<GL_LIGHTING> dlight;
          glColor3f(1,1,1);
          glViewport(0, 0, 128, 128);
          glMatrixMode(GL_PROJECTION);
          glPushMatrix();
          glLoadIdentity();
          glOrtho(0,1,0,1,-1,1);
          glMatrixMode(GL_MODELVIEW);
          glPushMatrix();{
          glLoadIdentity();
          glBegin(GL_QUADS);{
          glTexCoord2f(0,0);
          glVertex2f(0,0);
          glTexCoord2f(0,1);
          glVertex2f(0,1);
          glTexCoord2f(1,1);
          glVertex2f(1,1);
          glTexCoord2f(1,0);
          glVertex2f(1,0);
          }glEnd();
          }glPopMatrix();
          glMatrixMode(GL_PROJECTION);
          glPopMatrix();
          glMatrixMode(GL_MODELVIEW);
          glViewport(0, 0, GetWidth(), GetHeight());
          }
        */


        // Render the world and apply the shadow map texture to it
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(lastProjectionMatrix);
        glMatrixMode(GL_MODELVIEW);
        ApplyShadowMap();
        {
            // NICO
            Enable<GL_TEXTURE_2D> texture_on;
            glDisable(GL_BLEND);
            glBindTexture(GL_TEXTURE_2D, ShadowTextureMask);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, GetWidth(), GetHeight());
        }
        if (_background==0)
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        else if (_background==1)
            glClearColor(0.0f,0.0f,0.0f,0.0f);
        else if (_background==2)
            glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (_background==0)
            DrawLogo();

        DisplayOBJs();

        {
            float ofu = GetWidth()/(float)SHADOW_MASK_SIZE;
            float ofv = GetHeight()/(float)SHADOW_MASK_SIZE;
            glActiveTextureARB(GL_TEXTURE0_ARB);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D,ShadowTextureMask);
            //glTexEnvi(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,  GL_REPLACE);
            Disable<GL_DEPTH_TEST> dtoff;
            Disable<GL_LIGHTING> dlight;
            Enable<GL_BLEND> blend_on;
            glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            glColor3f(1,1,1);
            glViewport(0, 0, GetWidth(), GetHeight());
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0,1,0,1,-1,1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            {
                glLoadIdentity();
                glBegin(GL_QUADS);
                {
                    glTexCoord2f(0,0);
                    glVertex2f(0,0);
                    glTexCoord2f(0,ofv);
                    glVertex2f(0,1);
                    glTexCoord2f(ofu,ofv);
                    glVertex2f(1,1);
                    glTexCoord2f(ofu,0);
                    glVertex2f(1,0);
                } glEnd();
            } glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glViewport(0, 0, GetWidth(), GetHeight());
            glDisable(GL_TEXTURE_2D);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

    }
    else
#endif
    {
        if (_background==0)
            DrawLogo();


        camera()->getModelViewMatrix( lastModelviewMatrix );


        if (_renderingMode == GL_RENDER)
        {
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

            DisplayOBJs();

            DisplayMenu();		// always needs to be the last object being drawn
        }
    }
}

void QtGLViewer::viewAll()
{
    getSimulation()->computeBBox(groot, sceneMinBBox.ptr(), sceneMaxBBox.ptr());

    sceneBBoxIsValid = true;
    QGLViewer::setSceneBoundingBox(   qglviewer::Vec(sceneMinBBox.ptr()),qglviewer::Vec(sceneMaxBBox.ptr()) );

    qglviewer::Vec pos;
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 75.0;
    camera()->setPosition(pos);
    camera()->showEntireScene();
}

#ifdef SOFA_DEV

void QtGLViewer::DrawAutomate(void)
{

    std::cout << "DrawAutomate\n";


    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, 12.5, -10, 10, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for(int i = 0; i < (int) Automate::getInstance()->tabNodes.size(); i++)
    {
        for(int j = 0; j < (int) Automate::getInstance()->tabNodes[i]->tabOutputs.size(); j++)
        {
            Automate::getInstance()->tabNodes[i]->tabOutputs[j]->draw();
        }

        Automate::getInstance()->tabNodes[i]->draw();
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    update();
}

#endif // SOFA_DEV


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void QtGLViewer::resizeGL(int width, int height)
{
    _W = width;
    _H = height;


    QGLViewer::resizeGL( width,  height);
    camera()->setScreenWidthAndHeight(_W,_H);

    emit( resizeW( _W ) );
    emit( resizeH( _H ) );

}


// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::draw()
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    else if (_background==1)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    else if (_background==2)
        glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClearDepth(1.0);
    glClear(_clearBuffer);

#ifdef SOFA_DEV
    if (!_automateDisplayed)
    {
#endif // SOFA_DEV
        // draw the scene
        DrawScene();
#ifdef SOFA_DEV
    }
    else
    {
        DrawAutomate();
    }
#endif // SOFA_DEV


    if (_video)
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ((counter++ % CAPTURE_PERIOD)==0)
#endif
            screenshot(capture.findFilename());
    }

    if (_waitForRender)
        _waitForRender = false;

    emit( redrawn() );
}

// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------


void QtGLViewer::keyPressEvent ( QKeyEvent * e )
{
    // 	cerr<<"QtGLViewer::keyPressEvent, get "<<e->key()<<endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
        //cerr<<"QtGLViewer::keyPressEvent, key = "<<e->key()<<" with Control pressed "<<endl;
        sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
        groot->propagateEvent(&keyEvent);
    }
    else  // control the GUI
    {
        switch(e->key())
        {
        case Qt::Key_A: //axis
        case Qt::Key_S: //screenshot
        case Qt::Key_H: //shortcuts for screenshot and help page specified for qglviewer
        {
            QGLViewer::keyPressEvent(e);
            break;
        }
        case Qt::Key_T:
        {
            if (camera()->type() == qglviewer::Camera::ORTHOGRAPHIC)
                camera()->setType( qglviewer::Camera::PERSPECTIVE  );
            else
                camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
            break;
        }
        case Qt::Key_C:
        {
            viewAll();
            break;
        }
        default:
        {
            SofaViewer::keyPressEvent(e);
            QGLViewer::keyPressEvent(e);
            e->ignore();
        }
        }
    }
    update();
}


void QtGLViewer::keyReleaseEvent ( QKeyEvent * e )
{
    QGLViewer::keyReleaseEvent(e);
    SofaViewer::keyReleaseEvent(e);
}

void QtGLViewer::mousePressEvent ( QMouseEvent * e )
{
    if( ! mouseEvent(e) )
        QGLViewer::mousePressEvent(e);
}

void QtGLViewer::mouseReleaseEvent ( QMouseEvent * e )
{
    if( ! mouseEvent(e) )
        QGLViewer::mouseReleaseEvent(e);
}

void QtGLViewer::mouseMoveEvent ( QMouseEvent * e )
{
    if( ! mouseEvent(e) )
        QGLViewer::mouseMoveEvent(e);
}


bool QtGLViewer::mouseEvent(QMouseEvent * e)
{
    if(e->state()&Qt::ShiftButton)
    {
        SofaViewer::mouseEvent(e);
        return true;
    }
    else if (e->state()&Qt::ControlButton)
    {
        moveLaparoscopic(e);
        return true;
    }
    else
    {
        if (interactor!=NULL)
            interactor->newEvent("hide");
    }
    return false;
}

void QtGLViewer::moveRayPickInteractor(int eventX, int eventY)
{
    Vec3d p0, px, py, pz;
    gluUnProject(eventX, lastViewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(p0[0]), &(p0[1]), &(p0[2]));
    gluUnProject(eventX+1, lastViewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(px[0]), &(px[1]), &(px[2]));
    gluUnProject(eventX, lastViewport[3]-1-(eventY+1), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(py[0]), &(py[1]), &(py[2]));
    gluUnProject(eventX, lastViewport[3]-1-(eventY), 1, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(pz[0]), &(pz[1]), &(pz[2]));
    px -= p0;
    py -= p0;
    pz -= p0;
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

    //std::cout << p0[0]<<' '<<p0[1]<<' '<<p0[2] << " -> " << pz[0]<<' '<<pz[1]<<' '<<pz[2] << std::endl;
    interactor->newPosition(p0, q, transform);
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtGLViewer::resetView()
{

    viewAll();

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+".view";
        std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
            qglviewer::Vec pos;
            in >> pos[0];
            in >> pos[1];
            in >> pos[2];

            camera()->setPosition(pos);

            qglviewer::Quaternion q;

            in >> q[0];
            in >> q[1];
            in >> q[2];
            in >> q[3];
            q.normalize();

            camera()->setOrientation(q);
            camera()->showEntireScene();
            in.close();
            update();
            return;
        }
    }
    update();
}

#ifdef SOFA_DEV

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtGLViewer::SwitchToAutomateView()
{
    qglviewer::Vec pos;
    pos[0] = -10.0;
    pos[1] = 0.0;
    pos[2] = -50.0;
    camera()->setPosition(pos);
    qglviewer::Quaternion q;
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    camera()->setOrientation(q);
}

#endif // SOFA_DEV

void QtGLViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+".view";
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

#ifdef SOFA_DEV

/// Render Scene called during multiThread simulation using automate
void QtGLViewer::drawFromAutomate()
{
    update();
}

void QtGLViewer::automateDisplayVM(void)
{
    std::vector<core::VisualModel *>::iterator it = simulation::automatescheduler::ThreadSimulation::getInstance()->vmodels.begin();
    std::vector<core::VisualModel *>::iterator itEnd = simulation::automatescheduler::ThreadSimulation::getInstance()->vmodels.end();

    while (it != itEnd)
    {
        (*it)->draw();
        ++it;
    }
}

#endif // SOFA_DEV

void QtGLViewer::setSizeW( int size )
{
    resizeGL( size, _H );
    updateGL();
}

void QtGLViewer::setSizeH( int size )
{
    resizeGL( _W, size );
    updateGL();

}


QString QtGLViewer::helpString()
{

    QString text(
        "<H1>QtGLViewer</H1><hr>\
<ul>\
<li><b>Mouse</b>: TO NAVIGATE<br></li>\
<li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
<li><b>A</b>: TO DRAW AXIS<br></li>\
<li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
<li><b>C</b>: TO CENTER THE VIEW<br></li>\
<li><b>H</b>: TO OPEN HELP of QGLViewer<br></li>\
<li><b>L</b>: TO DRAW SHADOWS<br></li>\
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
<li><b>Esc</b>: TO QUIT ::sofa:: <br></li></ul>");

    return text;
}





} // namespace qgl

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa
