/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

SOFA_DECL_CLASS(OglViewport)
//Register OglViewport in the Object Factory
int OglViewportClass = core::RegisterObject("OglViewport")
        .add< OglViewport >()
        ;


OglViewport::OglViewport()
    :p_screenPosition(initData(&p_screenPosition, "screenPosition", "Viewport position"))
    ,p_screenSize(initData(&p_screenSize, "screenSize", "Viewport size"))
    ,p_cameraPosition(initData(&p_cameraPosition, Vec3f(0.0,0.0,0.0), "cameraPosition", "Camera's position in eye's space"))
    ,p_cameraOrientation(initData(&p_cameraOrientation,Quat(), "cameraOrientation", "Camera's orientation"))
    ,p_cameraRigid(initData(&p_cameraRigid, "cameraRigid", "Camera's rigid coord"))
    ,p_zNear(initData(&p_zNear, "zNear", "Camera's ZNear"))
    ,p_zFar(initData(&p_zFar, "zFar", "Camera's ZFar"))
    ,p_fovy(initData(&p_fovy, (double) 60.0, "fovy", "Field of View (Y axis)"))
    ,p_enabled(initData(&p_enabled, true, "enabled", "Enable visibility of the viewport"))
    ,p_advancedRendering(initData(&p_advancedRendering, false, "advancedRendering", "If true, viewport will be hidden if advancedRendering visual flag is not enabled"))
    ,p_useFBO(initData(&p_useFBO, true, "useFBO", "Use a FBO to render the viewport"))
    ,p_swapMainView(initData(&p_swapMainView, false, "swapMainView", "Swap this viewport with the main view"))
    ,p_drawCamera(initData(&p_drawCamera, false, "drawCamera", "Draw a frame representing the camera (see it in main viewport)"))
{
}

OglViewport::~OglViewport()
{
}


void OglViewport::init()
{
    if (p_cameraRigid.isSet() || p_cameraRigid.getParent())
    {
        p_cameraPosition.setDisplayed(false);
        p_cameraOrientation.setDisplayed(false);
    }
    else
    {
        p_cameraRigid.setDisplayed(false);
    }
}

void OglViewport::initVisual()
{
    if (p_useFBO.getValue())
    {
        const Vec<2, unsigned int> screenSize = p_screenSize.getValue();
        fbo.init(screenSize[0],screenSize[1]);
    }
}

bool OglViewport::isVisible(const core::visual::VisualParams*)
{
    if (!p_enabled.getValue())
        return false;
    if (p_advancedRendering.getValue())
    {
        VisualStyle* vstyle = NULL;
        this->getContext()->get(vstyle);
        if (vstyle && !vstyle->displayFlags.getValue().getShowRendering())
            return false;
    }
    return true;
}

void OglViewport::preDrawScene(core::visual::VisualParams* vp)
{
    if (!isVisible(vp)) return;

    if (p_swapMainView.getValue())
    {
        const sofa::defaulttype::BoundingBox& sceneBBox = vp->sceneBBox();
        Vec3f cameraPosition;
        Quat cameraOrientation;

        //Take the rigid if it is connected to something
        if (p_cameraRigid.isDisplayed())
        {
            RigidCoord rcam = p_cameraRigid.getValue();
            cameraPosition =  rcam.getCenter() ;
            cameraOrientation = rcam.getOrientation();
        }
        else
        {
            cameraPosition = p_cameraPosition.getValue();
            cameraOrientation = p_cameraOrientation.getValue();
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
        if (fabs(p_zNear.getValue()) < 0.0001 || fabs(p_zFar.getValue()) < 0.0001)
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
            zNear = p_zNear.getValue();
            zFar = p_zFar.getValue();
        }

        //Launch FBO process
        const Vec<2, unsigned int> screenSize( vp->viewport()[2], vp->viewport()[3] );
        double ratio = (double)screenSize[0]/(double)screenSize[1];
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(p_fovy.getValue(),ratio,zNear, zFar);

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

    if (p_swapMainView.getValue())
    {
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
    renderToViewport(vp);
    if (p_useFBO.getValue())
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
    const Vec<2, int> screenPosition = p_screenPosition.getValue();
    const Vec<2, unsigned int> screenSize = p_screenSize.getValue();
    int x0 = (screenPosition[0]>=0 ? screenPosition[0] : viewport[2]+screenPosition[0]);
    int y0 = (screenPosition[1]>=0 ? screenPosition[1] : viewport[3]+screenPosition[1]);
    if (p_useFBO.getValue())
    {
        fbo.init(screenSize[0],screenSize[1]);
        fbo.start();
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

    if (p_swapMainView.getValue())
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
        if (p_cameraRigid.isDisplayed())
        {
            RigidCoord rcam = p_cameraRigid.getValue();
            cameraPosition =  rcam.getCenter() ;
            cameraOrientation = rcam.getOrientation();
        }
        else
        {
            cameraPosition = p_cameraPosition.getValue();
            cameraOrientation = p_cameraOrientation.getValue();
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
        if (fabs(p_zNear.getValue()) < 0.0001 || fabs(p_zFar.getValue()) < 0.0001)
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
            zNear = p_zNear.getValue();
            zFar = p_zFar.getValue();
        }

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(p_fovy.getValue(),ratio,zNear, zFar);

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

    if (p_useFBO.getValue())
    {
        fbo.stop();
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
    if (!p_useFBO.getValue())
        return;

    const Viewport& viewport = vp->viewport();
    float vxmax, vymax;
    float vxmin, vymin;
    float txmax,tymax;
    float txmin,tymin;

    const Vec<2, int> screenPosition = p_screenPosition.getValue();
    const Vec<2, unsigned int> screenSize = p_screenSize.getValue();

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
    glBindTexture(GL_TEXTURE_2D, fbo.getColorTexture());

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

    //glViewport(0,0,vp->viewport[2],vp->viewport[3]);
}

void OglViewport::draw(const core::visual::VisualParams* vparams)
{
	if (!p_drawCamera.getValue())
		return;

	if (!p_cameraRigid.isDisplayed())
		vparams->drawTool()->drawFrame(p_cameraPosition.getValue(), p_cameraOrientation.getValue(), Vector3(0.1,0.1,0.1));
	else
	{
		RigidCoord rcam = p_cameraRigid.getValue();
		vparams->drawTool()->drawFrame(rcam.getCenter(), rcam.getOrientation(), Vector3(0.1,0.1,0.1));
	}
}

}

}

}
