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
#ifndef SOFA_GUI_QT_QTVIEWER_H
#define SOFA_GUI_QT_QTVIEWER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/system/gl.h>
#include <qgl.h>
#include <qtimer.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>


#include <viewer/SofaViewer.h>

#include <viewer/ViewerFactory.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/gl/VisualParameters.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/xml/Element.h>

// allow catheter navigation using the tracking system (very simple version, surely will be modified)
//#define TRACKING

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

//using namespace sofa::defaulttype;
using sofa::defaulttype::Vector3;
using sofa::defaulttype::Quaternion;
using namespace sofa::helper::gl;
using namespace sofa::helper::system::thread;
using namespace sofa::component::collision;


class QtViewer : public QGLWidget,  public sofa::gui::qt::viewer::OglModelSofaViewer
{
    Q_OBJECT

private:
#ifdef TRACKING
    double savedX;
    double savedY;
    bool firstTime;
    bool tracking;
#endif // TRACKING

    // Interaction
    enum
    {
        XY_TRANSLATION = 1,
        Z_TRANSLATION = 2,
    };

    enum { MINMOVE = 10 };

    VisualParameters visualParameters;

    QTimer* timerAnimate;
    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];

    Trackball		_currentTrackball;
    Trackball		_newTrackball;
    //	Quaternion		_currentQuat;
    //	Quaternion		_newQuat;
    int				_mouseX, _mouseY;
    int				_savedMouseX, _savedMouseY;

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

    ctime_t			_beginTime;

    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];

public:

    static const std::string VIEW_FILE_EXTENSION;


    static void create(QtViewer*& instance, const SofaViewerArgument& arg)
    {
        instance = new QtViewer(arg.parent, arg.name.c_str() );
    }

    static const char* viewerName()
    {
        return "OpenGL";
    }

    static const char* acceleratedName()
    {
        return "Open&GL";
    }

    /// Activate this class of viewer.
    /// This method is called before the viewer is actually created
    /// and can be used to register classes associated with in the the ObjectFactory.
    static int EnableViewer();

    /// Disable this class of viewer.
    /// This method is called after the viewer is destroyed
    /// and can be used to unregister classes associated with in the the ObjectFactory.
    static int DisableViewer();



    static QGLFormat setupGLFormat();
    QtViewer( QWidget* parent, const char* name="" );
    ~QtViewer();

    QWidget* getQWidget() { return this; }

    bool ready() {return _waitForRender;};
    void wait() {_waitForRender = true;};

public slots:
    void resetView();
    virtual void saveView();
    virtual void setSizeW(int);
    virtual void setSizeH(int);

    virtual void getView(Vec3d& pos, Quat& ori) const;
    virtual void setView(const Vec3d& pos, const Quat &ori);
    virtual void newView();
    virtual void moveView(const Vec3d& pos, const Quat &ori);
    virtual void captureEvent() { SofaViewer::captureEvent(); }
    virtual void drawColourPicking (core::CollisionModel::ColourCode code);

signals:
    void redrawn();
    void resizeW( int );
    void resizeH( int );
    void quit();


protected:

    void calcProjection();
    void initializeGL();
    void paintGL();
    void resizeGL( int w, int h );

public:

    sofa::simulation::Node* getScene()
    {
        return groot;
    }

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
    void moveRayPickInteractor(int eventX, int eventY);
    /////////////////
    // Interaction //
    /////////////////

    bool _mouseInteractorTranslationMode;
    bool _mouseInteractorRotationMode;
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


    QString helpString();

private:

    void	InitGFX(void);
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(SReal* minBBox, SReal* maxBBox, SReal r=0.0);
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
    void	DisplayOBJs();
    void	DisplayMenu(void);
    void	DrawScene();
    void  MakeStencilMask();

    void	ApplySceneTransformation(int x, int y);
    //int		handle(int event);	// required by FLTK

protected:
    //virtual bool event ( QEvent * e );

    virtual void keyPressEvent ( QKeyEvent * e );
    virtual void keyReleaseEvent ( QKeyEvent * e );
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void wheelEvent ( QWheelEvent* e);
    virtual bool mouseEvent ( QMouseEvent * e );
};

} // namespace qt

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa

#endif


