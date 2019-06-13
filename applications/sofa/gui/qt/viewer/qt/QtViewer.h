/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_QT_QTVIEWER_H
#define SOFA_GUI_QT_QTVIEWER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <qgl.h>
#include <qtimer.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <QtGlobal>

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
#include <QOpenGLWidget>
#include <QSurfaceFormat>
#include <QOpenGLContext>
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400

#include <sofa/gui/qt/viewer/SofaViewer.h>

#include "../../../ViewerFactory.h"

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Texture.h>

#include <sofa/helper/system/thread/CTime.h>
#include <SofaSimulationCommon/xml/Element.h>

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

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
typedef QOpenGLWidget QOpenGLWidget;
#else
typedef QGLWidget QOpenGLWidget;
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400

class QtViewer
        : public QOpenGLWidget
        , public sofa::gui::qt::viewer::OglModelSofaViewer
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


    QTimer* timerAnimate;
    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];

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


    static QtViewer* create(QtViewer*, BaseViewerArgument& arg)
    {
        BaseViewerArgument* pArg = &arg;
        ViewerQtArgument* viewerArg = dynamic_cast<ViewerQtArgument*>(pArg);
        return viewerArg ?
                new QtViewer(viewerArg->getParentWidget(), viewerArg->getName().c_str(), viewerArg->getNbMSAASamples() ) :
                new QtViewer(nullptr, pArg->getName().c_str(), pArg->getNbMSAASamples() )
                ;
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

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
    static QSurfaceFormat setupGLFormat(const unsigned int nbMSAASamples = 1);
#else
    static QGLFormat setupGLFormat(const unsigned int nbMSAASamples = 1);
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400
    QtViewer( QWidget* parent, const char* name="", const unsigned int nbMSAASamples = 1 );
    ~QtViewer() override;

    QWidget* getQWidget() override { return this; }

    bool ready() override {return !_waitForRender;}
    void wait() override {_waitForRender = true;}

public slots:
    void resetView() override;
    virtual void saveView() override;
    virtual void setSizeW(int) override;
    virtual void setSizeH(int) override;

    virtual void getView(defaulttype::Vector3& pos, defaulttype::Quat& ori) const override;
    virtual void setView(const defaulttype::Vector3& pos, const defaulttype::Quat &ori) override ;
    virtual void newView() override ;
    virtual void moveView(const defaulttype::Vector3& pos, const defaulttype::Quat &ori) override ;
    virtual void captureEvent()  override { SofaViewer::captureEvent(); }
    virtual void drawColourPicking (ColourPickingVisitor::ColourCode code) override ;
    virtual void fitNodeBBox(sofa::core::objectmodel::BaseNode * node )  override { SofaViewer::fitNodeBBox(node); }
    virtual void fitObjectBBox(sofa::core::objectmodel::BaseObject * obj)  override { SofaViewer::fitObjectBBox(obj); }

signals:
    void redrawn() override ;
    void resizeW( int ) override ;
    void resizeH( int ) override ;
    void quit();


protected:

    void calcProjection( int width = 0, int height = 0 );
    void initializeGL() override;
    void paintGL() override;
    void paintEvent(QPaintEvent* qpe) override;
    void resizeGL( int w, int h ) override;
    /// Overloaded from SofaViewer
    virtual void viewAll()  override {}

public:

    sofa::simulation::Node* getScene() override
    {
        return groot.get();
    }

    //void			reshape(int width, int height);
    int GetWidth()
    {
        return _W;
    }
    int GetHeight()
    {
        return _H;
    }

    void	UpdateOBJ(void);
    void moveRayPickInteractor(int eventX, int eventY) override ;
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
    static bool _mouseTrans;
    static bool _mouseRotate;


    QString helpString() const  override ;
//    void setCameraMode(core::visual::VisualParams::CameraType mode);
    void screenshot(const std::string& filename, int compression_level = -1) override;

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
    virtual void	drawScene() override ;
    void  MakeStencilMask();

    void	ApplySceneTransformation(int x, int y);
    //int		handle(int event);	// required by FLTK

    //virtual bool event ( QEvent * e );

    void keyPressEvent ( QKeyEvent * e ) override;
    void keyReleaseEvent ( QKeyEvent * e ) override;
    void mousePressEvent ( QMouseEvent * e ) override;
    void mouseReleaseEvent ( QMouseEvent * e ) override;
    void mouseMoveEvent ( QMouseEvent * e ) override;
    void wheelEvent ( QWheelEvent* e) override;
    virtual bool mouseEvent ( QMouseEvent * e ) override;
};

} // namespace qt

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa

#endif


