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
#ifndef SOFA_GUI_SIMPLE_SIMPLEGUI_H
#define SOFA_GUI_SIMPLE_SIMPLEGUI_H

#include <sofa/gui/BaseGUI.h>

#include <sofa/gui/PickHandler.h>

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/gl/Capture.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <SofaBaseVisual/InteractiveCamera.h>
#ifdef SOFA_SMP
#include <Multigraph.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>


#ifdef SOFA_BUILD_SOFAGUIGLUT
#	define SOFA_SOFAGUIGLUT_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_SOFAGUIGLUT_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa
{

namespace gui
{

namespace glut
{

using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using namespace sofa::helper::system::thread;
using namespace sofa::component::collision;
#ifdef SOFA_SMP
class MainLoopTask;
#endif

class SOFA_SOFAGUIGLUT_API SimpleGUI : public sofa::gui::BaseGUI
{

public:
    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DrawToolGL   DrawToolGL;

    /// @name methods each GUI must implement
    /// @{

    SimpleGUI();

    int mainLoop();
    void redraw();
    int closeGUI();

    sofa::simulation::Node* currentSimulation()
    {
        return getScene();
    }

    /// @}

    /// @name registration of each GUI
    /// @{

    static BaseGUI* CreateGUI(const char* name, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~SimpleGUI();

public:

    // glut callbacks

    static SimpleGUI* instance;
#ifdef SOFA_SMP
    Iterative::Multigraph<MainLoopTask> *mg;
#endif
    static void glut_display();
    static void glut_reshape(int w, int h);
    static void glut_keyboard(unsigned char k, int x, int y);
    static void glut_mouse(int button, int state, int x, int y);
    static void glut_motion(int x, int y);
    static void glut_special(int k, int x, int y);
    static void glut_idle();

private:

    enum
    {
        //TRACKBALL_MODE = 1,
        //PAN_MODE = 2,
        //ZOOM_MODE = 3,

        BTLEFT_MODE = 101,
        BTRIGHT_MODE = 102,
        BTMIDDLE_MODE = 103,
    };
    // Interaction
    enum
    {
        XY_TRANSLATION = 1,
        Z_TRANSLATION = 2,
    };

    enum { MINMOVE = 10 };


    sofa::simulation::Node::SPtr groot;
    std::string sceneFileName;
    sofa::component::visualmodel::BaseCamera::SPtr currentCamera;

    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];
    int				_navigationMode;
    int				_mouseX, _mouseY;
    int				_savedMouseX, _savedMouseY;
    bool			_spinning;
    bool			_moving;
    bool			_video;
    bool			_animationOBJ; int _animationOBJcounter;// save a succession of .obj indexed by _animationOBJcounter
    bool			_axis;
    int 			_background;
    float			_zoomSpeed;
    float			_panSpeed;
    //Transformation	_sceneTransform;
    Vector3			_previousEyePos;
    GLUquadricObj*	_arrow;
    GLUquadricObj*	_tube;
    GLUquadricObj*	_sphere;
    GLUquadricObj*	_disk;
    GLuint			_numOBJmodels;
    GLuint			_materialMode;
    GLboolean		_facetNormal;
    float			_zoom;
    int				_renderingMode;
    bool			_waitForRender;
    //GLuint			_logoTexture;
    Texture			*texLogo;
    ctime_t			_beginTime;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    GLint lastViewport[4];
    bool initTexturesDone;
    Capture capture;
public:

    void step();
    void animate();
    void playpause();
    void resetScene();
    void resetView();
    void saveView();

    void screenshot(int compression_level = -1);
    void exportOBJ(bool exportMTL=true);
    void dumpState(bool);
    void setExportGnuplot(bool);

    void initializeGL();
    void paintGL();
    void resizeGL( int w, int h );

    void keyPressEvent ( int k );
    void keyReleaseEvent ( int k );

    enum EventType
    {
        MouseButtonPress, MouseMove, MouseButtonRelease
    };
    void mouseEvent ( int type, int x, int y, int bt );

    void eventNewStep();

protected:

    void calcProjection();

public:
    void setScene(sofa::simulation::Node::SPtr scene, const char* filename=NULL, bool temporaryFile=false);
    sofa::simulation::Node* getScene()
    {
        return groot.get();
    }
    const std::string& getSceneFileName()
    {
        return sceneFileName;
    }
    void setCameraMode(core::visual::VisualParams::CameraType);
    void getView(Vec3d& pos, Quat& ori) const;
    void setView(const Vec3d& pos, const Quat &ori);
    void moveView(const Vec3d& pos, const Quat &ori);
    void newView();

    int GetWidth()
    {
        return _W;
    };
    int GetHeight()
    {
        return _H;
    };

    void	UpdateOBJ();

    /////////////////
    // Interaction //
    /////////////////

    PickHandler pick;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;

private:

    void	InitGFX();
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(SReal* minBBox, SReal* maxBBox, double r=0.0);
    void	DrawXYPlane(double zo, double xmin, double xmax, double ymin,
            double ymax, double step);
    void	DrawYZPlane(double xo, double ymin, double ymax, double zmin,
            double zmax, double step);
    void	DrawXZPlane(double yo, double xmin, double xmax, double zmin,
            double zmax, double step);

    void	DrawLogo();
    void	DisplayOBJs();
    void	DisplayMenu();
    void	DrawScene();

protected:
    bool isControlPressed() const;
    bool isShiftPressed() const;
    bool isAltPressed() const;
    bool m_isControlPressed;
    bool m_isShiftPressed;
    bool m_isAltPressed;
    void updateModifiers();
    bool m_dumpState;
    bool m_displayComputationTime;
    bool m_exportGnuplot;
    std::ofstream* m_dumpStateStream;
    VisualParams* vparams;
    DrawToolGL   drawTool;
};

} // namespace glut

} // namespace gui

} // namespace sofa

#endif
