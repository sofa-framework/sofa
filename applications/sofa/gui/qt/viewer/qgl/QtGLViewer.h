/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_QGLVIEWER_QTVIEWER_H
#define SOFA_GUI_QGLVIEWER_QTVIEWER_H

#include <sofa/helper/system/gl.h>
#include <qgl.h>
#include <qtimer.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

#include <sofa/gui/qt/viewer/SofaViewer.h>
#include <sofa/gui/ViewerFactory.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/Trackball.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/xml/Element.h>

#include <QGLViewer/qglviewer.h>
#include <QGLViewer/manipulatedFrame.h>

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

using helper::system::thread::ctime_t;
using sofa::defaulttype::Vector3;
using sofa::defaulttype::Vec3d;
using sofa::defaulttype::Quat;
using sofa::core::visual::VisualParams;
using sofa::core::objectmodel::BaseNode;
using sofa::core::objectmodel::BaseObject;

class SOFA_SOFAGUIQT_API QtGLViewer : public QGLViewer,
                                      public OglModelSofaViewer
{
    typedef Vector3::value_type Real;
    Q_OBJECT

public:
    static QtGLViewer* create(QtGLViewer*,
                              BaseViewerArgument& arg);

    QtGLViewer( QWidget* parent, const char* name="",
                                 const unsigned int nbMSAASamples = 1 );
    ~QtGLViewer();

    QWidget* getQWidget() { return this; }

    static const char* viewerName()  { return "QGLViewer"; }
    static const char* acceleratedName()  { return "&QGLViewer"; }
    virtual void drawColourPicking (ColourPickingVisitor::ColourCode code);

    int GetWidth() { return _W; }
    int GetHeight() { return _H; }
    bool ready() {return !_waitForRender;}
    void wait() {_waitForRender = true;}

    void UpdateOBJ(void);
    void moveRayPickInteractor(int eventX, int eventY);

    void setCameraMode(VisualParams::CameraType mode);

    QString helpString() const;

public slots:
    void resetView();
    void saveView();
    void setSizeW(int);
    void setSizeH(int);

    virtual void getView(Vec3d& pos, Quat& ori) const;
    virtual void setView(const Vec3d& pos, const Quat &ori);
    virtual void captureEvent() { SofaViewer::captureEvent(); }
    void fitObjectBBox(BaseObject* object) ;
    void fitNodeBBox(BaseNode* node) ;

signals:
    void redrawn();
    void resizeW( int );
    void resizeH( int );
    void quit();

protected:
    static QGLFormat setupGLFormat(const unsigned int nbMSAASamples = 1);

    void init();
    virtual void	drawScene();
    virtual void	DrawLogo(void);

    /// Overloaded from QGLViewer to render the scene
    virtual void draw();
    void resizeGL( int w, int h );

    /// Overloaded from SofaViewer
    virtual void viewAll();
    virtual void switchAxisViewing() ;
    virtual void toogleBoundingBoxDraw() ;

    /// Overloaded from QGLViewer
    virtual void keyPressEvent(QKeyEvent* e);
    virtual void keyReleaseEvent(QKeyEvent* e);
    virtual void mousePressEvent(QMouseEvent* e);
    virtual void mouseReleaseEvent(QMouseEvent* e);
    virtual void mouseMoveEvent(QMouseEvent* e);
    virtual void wheelEvent(QWheelEvent* e);

private:
    void	InitGFX(void);
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(Real* minBBox, Real* maxBBox, Real r=0.0);
    void	DrawXYPlane(double zo, double xmin, double xmax,
                        double ymin, double ymax, double step);
    void	DrawYZPlane(double xo, double ymin, double ymax,
                        double zmin, double zmax, double step);
    void	DrawXZPlane(double yo, double xmin, double xmax,
                        double zmin, double zmax, double step);
    void	CreateOBJmodelDisplayList(int material_mode);
    void	DisplayOBJs();
    void	DisplayMenu(void);
    void    MakeStencilMask();

private:
    QTimer*         timerAnimate;
    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];

    double          lastProjectionMatrix[16];
    double          lastModelviewMatrix[16];

    GLUquadricObj*	_arrow;
    GLUquadricObj*	_tube;
    GLUquadricObj*	_sphere;
    GLUquadricObj*	_disk;
    GLuint          _numOBJmodels;
    GLuint          _materialMode;
    GLboolean       _facetNormal;

    int             _renderingMode;
    ctime_t         _beginTime;

    bool            _waitForRender;
};

} // namespace qgl

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa

#endif


