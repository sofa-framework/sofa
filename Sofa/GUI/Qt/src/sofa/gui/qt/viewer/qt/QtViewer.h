/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gui/qt/config.h>

#include <sofa/gl/gl.h>
#include <sofa/gl/glu.h>
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

#include <sofa/gui/qt/viewer/OglModelPolicy.h>

#include <sofa/gui/common/ViewerFactory.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/helper/visual/Transformation.h>
#include <sofa/helper/visual/Trackball.h>
#include <sofa/gl/Texture.h>

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/xml/Element.h>

namespace sofa::gui::qt::viewer::qt
{

using sofa::type::Vec3;
using sofa::type::Quat;
using namespace sofa::gl;
using namespace sofa::helper::visual;
using namespace sofa::helper::system::thread;
using namespace sofa::component::collision;

#if defined(QT_VERSION) && QT_VERSION >= 0x050400
typedef QOpenGLWidget QOpenGLWidget;
#else
typedef QGLWidget QOpenGLWidget;
#endif // defined(QT_VERSION) && QT_VERSION >= 0x050400

class SOFA_GUI_QT_API QtViewer
        : public QOpenGLWidget
        , public sofa::gui::qt::viewer::OglModelSofaViewer
{
    Q_OBJECT

private:
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


    static QtViewer* create(QtViewer*, common::BaseViewerArgument& arg)
    {
        common::BaseViewerArgument* pArg = &arg;
        const common::ViewerQtArgument* viewerArg = dynamic_cast<common::ViewerQtArgument*>(pArg);
        return viewerArg ?
                new QtViewer(viewerArg->getParentWidget(), viewerArg->getName().c_str() ) :
                new QtViewer(nullptr, pArg->getName().c_str() )
                ;
    }

    static const char* viewerName()
    {
        return "OpenGL (QtViewer)";
    }

    static const char* acceleratedName()
    {
        return "Open&GL (QtViewer)";
    }

    /// Activate this class of viewer.
    /// This method is called before the viewer is actually created
    /// and can be used to register classes associated with in the ObjectFactory.
    static int EnableViewer();

    /// Disable this class of viewer.
    /// This method is called after the viewer is destroyed
    /// and can be used to unregister classes associated with in the ObjectFactory.
    static int DisableViewer();

    QtViewer( QWidget* parent, const char* name="");
    ~QtViewer() override;

    QWidget* getQWidget() override { return this; }

    bool ready() override {return !_waitForRender;}
    void wait() override {_waitForRender = true;}

public slots:
    void resetView() override;
    virtual void saveView() override;
    virtual void setSizeW(int) override;
    virtual void setSizeH(int) override;

    virtual void getView(type::Vec3& pos, type::Quat<SReal>& ori) const override;
    virtual void setView(const type::Vec3& pos, const type::Quat<SReal> &ori) override ;
    virtual void newView() override ;
    virtual void moveView(const type::Vec3& pos, const type::Quat<SReal> &ori) override ;
    virtual void captureEvent()  override { SofaViewer::captureEvent(); }
    virtual void drawColourPicking (common::ColourPickingVisitor::ColourCode code) override ;
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
    int getWidth() override
    {
        return _W;
    }
    int getHeight() override
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
    Quat<SReal> _mouseInteractorCurrentQuat;
    Vec3 _mouseInteractorAbsolutePosition;
    Trackball _mouseInteractorTrackball;
    void ApplyMouseInteractorTransformation(int x, int y);

    static Quat<SReal> _mouseInteractorNewQuat;
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

} // namespace sofa::gui::qt::viewer::qt
