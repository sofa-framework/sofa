/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <SofaOpenglVisual/OglViewport.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/system/glu.h>
#include <SofaBaseVisual/VisualStyle.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

//Register OglViewport in the Object Factory
int OglViewportClass = core::RegisterObject("OglViewport")
        .add< OglViewport >()
        ;


OglViewport::OglViewport()
    :d_screenPosition(initData(&d_screenPosition, "screenPosition", "Viewport position"))
    ,d_screenSize(initData(&d_screenSize, "screenSize", "Viewport size"))
    ,d_cameraPosition(initData(&d_cameraPosition, Vec3f(0.0,0.0,0.0), "cameraPosition", "Camera's position in eye's space"))
    ,d_cameraOrientation(initData(&d_cameraOrientation,Quat(), "cameraOrientation", "Camera's orientation"))
    ,d_cameraRigid(initData(&d_cameraRigid, "cameraRigid", "Camera's rigid coord"))
    ,d_zNear(initData(&d_zNear, "zNear", "Camera's ZNear"))
    ,d_zFar(initData(&d_zFar, "zFar", "Camera's ZFar"))
    ,d_fovy(initData(&d_fovy, (double) 60.0, "fovy", "Field of View (Y axis)"))
    ,d_enabled(initData(&d_enabled, true, "enabled", "Enable visibility of the viewport"))
    ,d_advancedRendering(initData(&d_advancedRendering, false, "advancedRendering", "If true, viewport will be hidden if advancedRendering visual flag is not enabled"))
    ,d_useFBO(initData(&d_useFBO, true, "useFBO", "Use a FBO to render the viewport"))
    ,d_swapMainView(initData(&d_swapMainView, false, "swapMainView", "Swap this viewport with the main view"))
    ,d_drawCamera(initData(&d_drawCamera, false, "drawCamera", "Draw a frame representing the camera (see it in main viewport)"))
{
}

OglViewport::~OglViewport()
{
}


void OglViewport::init()
{
    if (d_cameraRigid.isSet() || d_cameraRigid.getParent())
    {
        d_cameraPosition.setDisplayed(false);
        d_cameraOrientation.setDisplayed(false);
    }
    else
    {
        d_cameraRigid.setDisplayed(false);
    }
}

void OglViewport::initVisual()
{
    if (d_useFBO.getValue())
    {
        const Vec<2, unsigned int> screenSize = d_screenSize.getValue();
        fbo = std::unique_ptr<helper::gl::FrameBufferObject>(new helper::gl::FrameBufferObject());
        fbo->init(screenSize[0],screenSize[1]);
    }
}

bool OglViewport::isVisible(const core::visual::VisualParams*)
{
    if (!d_enabled.getValue())
        return false;
    if (d_advancedRendering.getValue())
    {
        VisualStyle* vstyle = NULL;
        this->getContext()->get(vstyle);
        if (vstyle && !vstyle->displayFlags.getValue().getShowAdvancedRendering())
            return false;
    }
    return true;
}

void OglViewport::preDrawScene(core::visual::VisualParams* vp)
{
    if (!isVisible(vp)) return;

    if (d_swapMainView.getValue())
    {
        const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
        Vec3f cameraPosition;
        Quat cameraOrientation;

        //Take the rigid if it is connected to something
        if (d_cameraRigid.isDisplayed())
        {
            RigidCoord rcam = d_cameraRigid.getValue();
            cameraPosition =  rcam.getCenter() ;
            cameraOrientation = rcam.getOrientation();
        }
        else
        {
            cameraPosition = d_cameraPosition.getValue();
            cameraOrientation = d_cameraOrientation.getValue();
        }

        cameraOrientation.normalize();
        helper::gl::Transformation transform;

        cameraOrientation.buildRotationMatrix(transform.rotation);
        //cameraOrientation.writeOpenGlMatrix((SReal*) transform.rotation);

        for (unsigned int i=0 ; i< 3 ; i++)
        {
            transform.translation[i] = -cameraPosition[i];
            transform.scale[i] = 1.0;
        }

        double zNear=1e10, zFar=-1e10;
        //recompute zNear, zFar
        if (fabs(d_zNear.getValue()) < 0.0001 || fabs(d_zFar.getValue()) < 0.0001)
        {
            for (int corner=0; corner<8; ++corner)
            {
                Vector3 p(
                    (corner&1)?sceneBBox.minBBox().x():sceneBBox.maxBBox().x(),
                    (corner&2)?sceneBBox.minBBox().y():sceneBBox.maxBBox().y(),
                    (corner&4)?sceneBBox.minBBox().z():sceneBBox.maxBBox().z()
                );
                //TODO: invert transform...
                p = transform * p;
                double z = -p[2];
                if (z < zNear) zNear = z;
                if (z > zFar)  zFar = z;
            }

            if (zNear <= 0)
                zNear = 1;
            if (zFar >= 1000.0)
                zFar = 1000.0;

            if (zNear > 0 && zFar < 1000)
            {
                zNear *= 0.8; // add some margin
                zFar *= 1.2;
                if (zNear < zFar*0.01)
                    zNear = zFar*0.01;
                if (zNear < 0.1) zNear = 0.1;
                if (zFar < 2.0) zFar = 2.0;
            }
        }
        else
        {
            zNear = d_zNear.getValue();
            zFar = d_zFar.getValue();
        }

        //Launch FBO process
        const Vec<2, unsigned int> screenSize( vp->viewport()[2], vp->viewport()[3] );
        double ratio = (double)screenSize[0]/(double)screenSize[1];
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(d_fovy.getValue(),ratio,zNear, zFar);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        helper::gl::glMultMatrix((SReal *)transform.rotation);

        helper::gl::glTranslate(transform.translation[0], transform.translation[1], transform.translation[2]);
    }
}

bool OglViewport::drawScene(core::visual::VisualParams* /* vp */)
{
    return false;
}

void OglViewport::postDrawScene(core::visual::VisualParams* vp)
{
    if (!isVisible(vp)) return;

    if (d_swapMainView.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
    renderToViewport(vp);
    if (d_useFBO.getValue())
        renderFBOToScreen(vp);
}

void OglViewport::renderToViewport(core::visual::VisualParams* vp)
{
    const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
    helper::gl::Transformation vp_sceneTransform = vp->sceneTransform();
//    double vp_zNear = vp->zNear();
//    double vp_zFar = vp->zFar();

    const Viewport viewport = vp->viewport();
    //Launch FBO process
    const Vec<2, int> screenPosition = d_screenPosition.getValue();
    const Vec<2, unsigned int> screenSize = d_screenSize.getValue();
    int x0 = (screenPosition[0]>=0 ? screenPosition[0] : viewport[2]+screenPosition[0]);
    int y0 = (screenPosition[1]>=0 ? screenPosition[1] : viewport[3]+screenPosition[1]);
    if (d_useFBO.getValue())
    {
        fbo->init(screenSize[0],screenSize[1]);
        fbo->start();
        glViewport(0,0,screenSize[0],screenSize[1]);
    }
    else
    {
        glViewport(x0,y0,screenSize[0],screenSize[1]);
        glScissor(x0,y0,screenSize[0],screenSize[1]);
        glEnable(GL_SCISSOR_TEST);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    double ratio = (double)screenSize[0]/(double)screenSize[1];

    if (d_swapMainView.getValue())
    {
        GLdouble matrix[16];
        glGetDoublev(GL_PROJECTION_MATRIX, matrix);
        double mainRatio = (double)viewport[2]/(double)viewport[3];
        double scale = mainRatio/ratio;

        matrix[0] *= scale;
        matrix[4] *= scale;
        matrix[8] *= scale;
        matrix[12] *= scale;

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixd(matrix);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
    }
    else
    {
        helper::gl::Transformation transform;
        double zNear=1e10, zFar=-1e10;
//        double fovy = 0;

        Vec3f cameraPosition;
        Quat cameraOrientation;

        //Take the rigid if it is connected to something
        if (d_cameraRigid.isDisplayed())
        {
            RigidCoord rcam = d_cameraRigid.getValue();
            cameraPosition =  rcam.getCenter() ;
            cameraOrientation = rcam.getOrientation();
        }
        else
        {
            cameraPosition = d_cameraPosition.getValue();
            cameraOrientation = d_cameraOrientation.getValue();
        }

        cameraOrientation.normalize();

        cameraOrientation.buildRotationMatrix(transform.rotation);
        //cameraOrientation.writeOpenGlMatrix((SReal*) transform.rotation);

        for (unsigned int i=0 ; i< 3 ; i++)
        {
            transform.translation[i] = -cameraPosition[i];
            transform.scale[i] = 1.0;
        }

        //recompute zNear, zFar
        if (fabs(d_zNear.getValue()) < 0.0001 || fabs(d_zFar.getValue()) < 0.0001)
        {
            for (int corner=0; corner<8; ++corner)
            {
                Vector3 p(
                    (corner&1)?sceneBBox.minBBox().x():sceneBBox.maxBBox().x(),
                    (corner&2)?sceneBBox.minBBox().y():sceneBBox.maxBBox().y(),
                    (corner&4)?sceneBBox.minBBox().z():sceneBBox.maxBBox().z()
                );
                //TODO: invert transform...
                p = transform * p;
                double z = -p[2];
                if (z < zNear) zNear = z;
                if (z > zFar)  zFar = z;
            }

            if (zNear <= 0)
                zNear = 1;
            if (zFar >= 1000.0)
                zFar = 1000.0;

            if (zNear > 0 && zFar < 1000)
            {
                zNear *= 0.8; // add some margin
                zFar *= 1.2;
                if (zNear < zFar*0.01)
                    zNear = zFar*0.01;
                if (zNear < 0.1) zNear = 0.1;
                if (zFar < 2.0) zFar = 2.0;
            }
        }
        else
        {
            zNear = d_zNear.getValue();
            zFar = d_zFar.getValue();
        }

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(d_fovy.getValue(),ratio,zNear, zFar);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        helper::gl::glMultMatrix((SReal *)transform.rotation);

        helper::gl::glTranslate(transform.translation[0], transform.translation[1], transform.translation[2]);

        //gluLookAt(cameraPosition[0], cameraPosition[1], cameraPosition[2],0 ,0 ,0, cameraDirection[0], cameraDirection[1], cameraDirection[2]);
    }
    vp->viewport() = Viewport(x0,y0,screenSize[0],screenSize[1]);
    vp->pass() = core::visual::VisualParams::Std;
    simulation::VisualDrawVisitor vdv( vp );
    vdv.setTags(this->getTags());
    vdv.execute ( getContext() );
    vp->pass() = core::visual::VisualParams::Transparent;
    simulation::VisualDrawVisitor vdvt( vp );
    vdvt.setTags(this->getTags());
    vdvt.execute ( getContext() );

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    if (d_useFBO.getValue())
    {
        fbo->stop();
    }
    else
    {
        glDisable(GL_SCISSOR_TEST);
    }
    vp->viewport() = viewport;
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
}

void OglViewport::renderFBOToScreen(core::visual::VisualParams* vp)
{
    if (!d_useFBO.getValue())
        return;

    const Viewport& viewport = vp->viewport();
    float vxmax, vymax;
    float vxmin, vymin;
    float txmax,tymax;
    float txmin,tymin;

    const Vec<2, int> screenPosition = d_screenPosition.getValue();
    const Vec<2, unsigned int> screenSize = d_screenSize.getValue();

    int x0 = (screenPosition[0]>=0 ? screenPosition[0] : viewport[2]+screenPosition[0]);
    int y0 = (screenPosition[1]>=0 ? screenPosition[1] : viewport[3]+screenPosition[1]);

    txmin = tymin = 0.0;
    txmax = tymax = 1.0;
    //glViewport(x0,y0,screenSize[0],screenSize[1]);
    vxmin = x0*2.0f/viewport[2] - 1.0f;
    vymin = y0*2.0f/viewport[3] - 1.0f;
    vxmax = (x0+screenSize[0])*2.0f/viewport[2] - 1.0f;
    vymax = (y0+screenSize[1])*2.0f/viewport[3] - 1.0f;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbo->getColorTexture());

    glBegin(GL_QUADS);
    {
        glColor3f(1,1,1);
        glTexCoord3f(txmin,tymax,0.0); glVertex3f(vxmin,vymax,0.0);
        glTexCoord3f(txmax,tymax,0.0); glVertex3f(vxmax,vymax,0.0);
        glTexCoord3f(txmax,tymin,0.0); glVertex3f(vxmax,vymin,0.0);
        glTexCoord3f(txmin,tymin,0.0); glVertex3f(vxmin,vymin,0.0);
    }
    glEnd();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void OglViewport::draw(const core::visual::VisualParams* vparams)
{
	if (!d_drawCamera.getValue())
		return;

	if (!d_cameraRigid.isDisplayed())
		vparams->drawTool()->drawFrame(d_cameraPosition.getValue(), d_cameraOrientation.getValue(), Vector3(0.1,0.1,0.1));
	else
	{
		RigidCoord rcam = d_cameraRigid.getValue();
		vparams->drawTool()->drawFrame(rcam.getCenter(), rcam.getOrientation(), Vector3(0.1,0.1,0.1));
	}
}

}

}

}
