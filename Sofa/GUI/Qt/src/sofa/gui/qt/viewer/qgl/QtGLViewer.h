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

#include <qtimer.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>

#include <sofa/gui/qt/viewer/OglModelPolicy.h>
#include <sofa/gui/common/ViewerFactory.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/helper/visual/Transformation.h>
#include <sofa/helper/visual/Trackball.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/gl/Texture.h>
#include <sofa/simulation//common/xml/Element.h>



#include <QGLViewer/qglviewer.h>
#include <QGLViewer/manipulatedFrame.h>

#define TRACKING_MOUSE

namespace sofa::gui::qt::viewer::qgl
{

class SOFA_GUI_QT_API QtGLViewer :public QGLViewer,   public sofa::gui::qt::viewer::OglModelSofaViewer
{
    typedef type::Vec3::value_type Real;
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

    GLUquadricObj*	_arrow;
    GLUquadricObj*	_tube;
    GLUquadricObj*	_sphere;
    GLUquadricObj*	_disk;
    GLuint _numOBJmodels;
    GLuint _materialMode;
    GLboolean _facetNormal;

    int _renderingMode;

    helper::system::thread::ctime_t _beginTime;


    bool _waitForRender;


public:

    static QtGLViewer* create(QtGLViewer*, sofa::gui::common::BaseViewerArgument& arg)
    {
        common::BaseViewerArgument* pArg = &arg;
        const common::ViewerQtArgument* viewerArg = dynamic_cast<common::ViewerQtArgument*>(pArg);
        return viewerArg ?
                new QtGLViewer(viewerArg->getParentWidget(), viewerArg->getName().c_str() ) :
                new QtGLViewer(nullptr, pArg->getName().c_str() )
                ;
    }

    static const char* viewerName()  { return "QGLViewer (QtGLViewer)"; }

    static const char* acceleratedName()  { return "&QGLViewer (QtGLViewer)"; }

    virtual void drawColourPicking (common::ColourPickingVisitor::ColourCode code) override;

    QtGLViewer( QWidget* parent, const char* name="");
    ~QtGLViewer() override;

    QWidget* getQWidget() override { return this; }

protected:

    //     void calcProjection();
    void init() override;
    /// Overloaded from QGLViewer to render the scene
    void draw() override;
    /// Overloaded from SofaViewer
    virtual void viewAll() override;
    void resizeGL( int w, int h ) override;

public:

    //void			reshape(int width, int height);
    int getWidth() override
    {
        return _W;
    }
    int getHeight() override
    {
        return _H;
    }
    bool ready() override {return !_waitForRender;}
    void wait() override {_waitForRender = true;}

    void	UpdateOBJ(void);

    void moveRayPickInteractor(int eventX, int eventY) override;

    void setCameraMode(core::visual::VisualParams::CameraType mode) override;

    void screenshot(const std::string& filename, int compression_level = -1) override;

    QString helpString() const override;


private:

    void	InitGFX(void);
    void	PrintString(void* font, char* string);
    void	Display3DText(float x, float y, float z, char* string);
    void	DrawAxis(double xpos, double ypos, double zpos, double arrowSize);
    void	DrawBox(SReal* minBBox, SReal* maxBBox, SReal r=0.0);
    void	DrawXYPlane(double zo, double xmin, double xmax, double ymin, double ymax, double step);
    void	DrawYZPlane(double xo, double ymin, double ymax, double zmin, double zmax, double step);
    void	DrawXZPlane(double yo, double xmin, double xmax, double zmin, double zmax, double step);
    void	CreateOBJmodelDisplayList(int material_mode);
    //int     loadBMP(char *filename, TextureImage *texture);
    //void	LoadGLTexture(char *Filename);
    void	DisplayOBJs();
    void	DisplayMenu(void);
    void        MakeStencilMask();

    //int		handle(int event);	// required by FLTK

protected:
    //virtual bool event ( QEvent * e );

    virtual void	drawScene() override;
    virtual void	DrawLogo(void);


    void keyPressEvent ( QKeyEvent * e ) override;
    void keyReleaseEvent ( QKeyEvent * e ) override;
    void mousePressEvent ( QMouseEvent * e ) override;
    void mouseReleaseEvent ( QMouseEvent * e ) override;
    void mouseMoveEvent ( QMouseEvent * e ) override;
    void wheelEvent(QWheelEvent* e) override;
    bool mouseEvent( QMouseEvent * e ) override;



public slots:
    void resetView() override;
    void saveView() override;
    void setSizeW(int) override;
    void setSizeH(int) override;

    virtual void getView(type::Vec3& pos, type::Quat<SReal>& ori) const override;
    virtual void setView(const type::Vec3& pos, const type::Quat<SReal> &ori) override;
    virtual void captureEvent() override { SofaViewer::captureEvent(); }
    void fitObjectBBox(sofa::core::objectmodel::BaseObject* object) override
    {
        if( object->f_bbox.getValue().isValid() && !object->f_bbox.getValue().isFlat() )
            this->camera()->fitBoundingBox(
                ::qglviewer::Vec(object->f_bbox.getValue().minBBox()),
                ::qglviewer::Vec(object->f_bbox.getValue().maxBBox())
            );
        else
        {
            if(object->getContext()->f_bbox.getValue().isValid() && !object->getContext()->f_bbox.getValue().isFlat()  )
            {
                this->camera()->fitBoundingBox(
                    ::qglviewer::Vec(object->getContext()->f_bbox.getValue().minBBox()),
                    ::qglviewer::Vec(object->getContext()->f_bbox.getValue().maxBBox())
                );
            }
        }
        this->update();
    }

    void fitNodeBBox(sofa::core::objectmodel::BaseNode* node) override
    {
        if( node->f_bbox.getValue().isValid() && !node->f_bbox.getValue().isFlat() )
            this->camera()->fitBoundingBox(
                ::qglviewer::Vec(node->f_bbox.getValue().minBBox()),
                ::qglviewer::Vec(node->f_bbox.getValue().maxBBox())
            );

        this->update();

    }

signals:
    void redrawn() override;
    void resizeW( int ) override;
    void resizeH( int ) override;
    void quit();
};

} // namespace sofa::gui::qt::viewer::qgl
