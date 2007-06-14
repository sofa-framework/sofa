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
#ifndef SOFA_GUI_QT_QTVIEWER_H
#define SOFA_GUI_QT_QTVIEWER_H

#include <qgl.h>
#include <qtimer.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>


#include "SofaViewer.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/gl/Capture.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/collision/RayPickInteractor.h>
#include <sofa/simulation/tree/xml/Element.h>
#include <sofa/simulation/automatescheduler/Automate.h>

namespace sofa
{

namespace gui
{

namespace qt
{

using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using namespace sofa::helper::system::thread;
using namespace sofa::simulation::automatescheduler;
using namespace sofa::component::collision;


class QtViewer :public QGLWidget,  public sofa::gui::viewer::SofaViewer//public Automate::DrawCB
{
    Q_OBJECT

private:

    enum
    {
        TRACKBALL_MODE = 1,
        PAN_MODE = 2,
        ZOOM_MODE = 3,

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


    QTimer* timerAnimate;
    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];
    int				_navigationMode;
    Trackball		_currentTrackball;
    Trackball		_newTrackball;
    //	Quaternion		_currentQuat;
    //	Quaternion		_newQuat;
    int				_mouseX, _mouseY;
    int				_savedMouseX, _savedMouseY;
    bool			_spinning;
    bool			_moving;
    bool			_video;

    bool			_axis;
    int 			_background;
    bool			_shadow;
    bool			_glshadow;
    float			_zoomSpeed;
    float			_panSpeed;
    Transformation	_sceneTransform;
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

    bool _waitForRender;


    //GLuint			_logoTexture;
    Texture			*texLogo;
    bool			_automateDisplayed;
    ctime_t			_beginTime;
    RayPickInteractor* interactor;
    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];
    GLint lastViewport[4];
    bool    sceneBBoxIsValid;
    Vector3 sceneMinBBox;
    Vector3 sceneMaxBBox;
    bool initTexturesDone;
    Capture capture;
public:

    /// Activate this class of viewer.
    /// This method is called before the viewer is actually created
    /// and can be used to register classes associated with in the the ObjectFactory.
    static int EnableViewer();

    /// Disable this class of viewer.
    /// This method is called after the viewer is destroyed
    /// and can be used to unregister classes associated with in the the ObjectFactory.
    static int DisableViewer();

    QtViewer( QWidget* parent, const char* name="" );
    ~QtViewer();

    QWidget* getQWidget() { return this; }

    bool ready() {return _waitForRender;};
    void wait() {_waitForRender = true;};

public slots:
    virtual void resetView();
    virtual void saveView();
    virtual void screenshot();
    virtual void setSizeW(int);
    virtual void setSizeH(int);

signals:
    void redrawn();
    void resizeW( int );
    void resizeH( int );



protected:

    void calcProjection();
    void initializeGL();
    void paintGL();
    void resizeGL( int w, int h );
    void ApplyShadowMap();
    void CreateRenderTexture(GLuint& textureID, int sizeX, int sizeY, int channels, int type);
    void StoreLightMatrices();

public:
    void setScene(sofa::simulation::tree::GNode* scene, const char* filename=NULL);
    sofa::simulation::tree::GNode* getScene()
    {
        return groot;
    }
    void			SwitchToPresetView();
    void			SwitchToAutomateView();
    //void			reshape(int width, int height);
    int GetWidth()
    {
        return _W;
    };
    int GetHeight()
    {
        return _H;
    };

    void	UpdateOBJ(void);

    /////////////////
    // Interaction //
    /////////////////

    bool _mouseInteractorTranslationMode;
    bool _mouseInteractorRotationMode;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;
    int _translationMode;
    Quaternion _mouseInteractorCurrentQuat;
    Vector3 _mouseInteractorAbsolutePosition;
    Trackball _mouseInteractorTrackball;
    void ApplyMouseInteractorTransformation(int x, int y);

    static Quaternion _mouseInteractorNewQuat;
    static Vector3 _mouseInteractorRelativePosition;
    static Quaternion _newQuat;
    static Quaternion _currentQuat;
    static bool _mouseTrans;
    static bool _mouseRotate;

    // Display scene from the automate
    void drawFromAutomate();
    static void	automateDisplayVM(void);
    QString helpString();
private:

    void	InitGFX(void);
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(double* minBBox, double* maxBBox, double r=0.0);
    void	DrawXYPlane(double zo, double xmin, double xmax, double ymin,
            double ymax, double step);
    void	DrawYZPlane(double xo, double ymin, double ymax, double zmin,
            double zmax, double step);
    void	DrawXZPlane(double yo, double xmin, double xmax, double zmin,
            double zmax, double step);
    void	CreateOBJmodelDisplayList(int material_mode);
    //int     loadBMP(char *filename, TextureImage *texture);
    //void	LoadGLTexture(char *Filename);
    void	DrawLogo(void);
    void	DisplayOBJs(bool shadowPass = false);
    void	DisplayMenu(void);
    void	DrawScene();
    void	DrawAutomate();
    void	ApplySceneTransformation(int x, int y);
    //int		handle(int event);	// required by FLTK

protected:
    //virtual bool event ( QEvent * e );
    bool m_isControlPressed;
    bool isControlPressed() const;

    virtual void keyPressEvent ( QKeyEvent * e );
    virtual void keyReleaseEvent ( QKeyEvent * e );
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void mouseEvent ( QMouseEvent * e );
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif


