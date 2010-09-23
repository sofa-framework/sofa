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



#include <QGLViewer/qglviewer.h>

#define TRACKING_MOUSE

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
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using namespace sofa::helper::system::thread;
using namespace sofa::component::collision;


class QtGLViewer :public QGLViewer,   public sofa::gui::qt::viewer::SofaViewer
{
    typedef Vector3::value_type Real;
    Q_OBJECT
private:

#ifdef TRACKING_MOUSE
    bool m_grabActived;
#endif

    QTimer* timerAnimate;
    int				_W, _H;
    int				_clearBuffer;
    bool			_lightModelTwoSides;
    float			_lightPosition[4];


    double lastProjectionMatrix[16];
    double lastModelviewMatrix[16];

    VisualParameters visualParameters;



    //     float			_zoomSpeed;
    //     float			_panSpeed;
    //     Transformation	_sceneTransform;
    //     Vector3			_previousEyePos;
    GLUquadricObj*	_arrow;
    GLUquadricObj*	_tube;
    GLUquadricObj*	_sphere;
    GLUquadricObj*	_disk;
    GLuint			_numOBJmodels;
    GLuint			_materialMode;
    GLboolean		_facetNormal;
    //     float			_zoom;
    int				_renderingMode;
    //GLuint			_logoTexture;
    Texture			*texLogo;

    ctime_t			_beginTime;


    bool _waitForRender;

    /*
              viewerOGREAction->setIconText(QApplication::translate("GUI", "OGRE", 0, QApplication::UnicodeUTF8));
            viewerOGREAction->setText(QApplication::translate("GUI", "&OGRE", 0, QApplication::UnicodeUTF8));
    */
public:

    static void create(QtGLViewer*& instance, const CreatorArgument& arg)
    {
        instance = new QtGLViewer(arg.parent, arg.name.c_str() );
    }

    virtual const char* getViewerName() const { return "QGLViewer"; }

    virtual const char* getAcceleratedViewerName() const { return "&QGLViewer"; }

    /// Activate this class of viewer.
    /// This method is called before the viewer is actually created
    /// and can be used to register classes associated with in the the ObjectFactory.
    static int EnableViewer();

    /// Disable this class of viewer.
    /// This method is called after the viewer is destroyed
    /// and can be used to unregister classes associated with in the the ObjectFactory.
    static int DisableViewer();

    void RegisterVisualModels() ;

    void UnregisterVisualModels() ;

    virtual void drawColourPicking (core::CollisionModel::ColourCode code);

    QtGLViewer( QWidget* parent, const char* name="" );
    ~QtGLViewer();

    QWidget* getQWidget() { return this; }

protected:

    //     void calcProjection();
    void init();
    virtual void draw();
    void viewAll();
    void resizeGL( int w, int h );

public:

    //void			reshape(int width, int height);
    int GetWidth()
    {
        return _W;
    };
    int GetHeight()
    {
        return _H;
    };
    bool ready() {return _waitForRender;};
    void wait() {_waitForRender = true;};

    void	UpdateOBJ(void);

    void moveRayPickInteractor(int eventX, int eventY);

    void setCameraMode(component::visualmodel::BaseCamera::CameraType mode);

    QString helpString();

    virtual void setBackgroundImage(std::string imageFileName);



private:

    void	InitGFX(void);
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(Real* minBBox, Real* maxBBox, Real r=0.0);
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

    //int		handle(int event);	// required by FLTK

protected:
    //virtual bool event ( QEvent * e );



    virtual void keyPressEvent ( QKeyEvent * e );
    virtual void keyReleaseEvent ( QKeyEvent * e );
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void wheelEvent(QWheelEvent* e);
    bool mouseEvent( QMouseEvent * e );




public slots:
    void resetView();
    void saveView();
    void setSizeW(int);
    void setSizeH(int);
    virtual void captureEvent() { SofaViewer::captureEvent(); }

signals:
    void redrawn();
    void resizeW( int );
    void resizeH( int );
    void quit();
};

} // namespace qgl

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa

#endif


