/*
 * OglViewport.cpp
 *
 *  Created on: 26 nov. 2009
 *      Author: froy
 */

#include "OglViewport.h"
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/gl/template.h>

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
{
    // TODO Auto-generated constructor stub

}

OglViewport::~OglViewport()
{
    // TODO Auto-generated destructor stub
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
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GLint windowWidth = viewport[2];
    GLint windowHeight = viewport[3];

    fbo.init(windowWidth,windowHeight);
}


void OglViewport::preDrawScene(helper::gl::VisualParameters* vp)
{
    Vec3f cameraPosition;
    Quat cameraOrientation;
    //const Vec3f &cameraPosition = *p_cameraPosition.beginEdit();
    //const Quat &cameraOrientation = *p_cameraOrientation.beginEdit();
    //defaulttype::RigidCoord &rigidCamera = *p_cameraRigid.beginEdit();

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
            Vector3 p((corner&1)?vp->minBBox[0]:vp->maxBBox[0],
                    (corner&2)?vp->minBBox[1]:vp->maxBBox[1],
                    (corner&4)?vp->minBBox[2]:vp->maxBBox[2]);
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

    double ratio = (double)vp->viewport[2]/(double)vp->viewport[3];

    //Launch FBO process
    fbo.setSize(vp->viewport[2], vp->viewport[3]);
    fbo.start();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

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
    simulation::VisualDrawVisitor vdv( core::ExecParams::defaultInstance() /* PARAMS FIRST */, core::VisualModel::Std );
    vdv.execute ( getContext() );
    simulation::VisualDrawVisitor vdvt( core::ExecParams::defaultInstance() /* PARAMS FIRST */, core::VisualModel::Transparent );
    vdvt.execute ( getContext() );

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    fbo.stop();
}

bool OglViewport::drawScene(helper::gl::VisualParameters* /* vp */)
{
    return false;
}

void OglViewport::postDrawScene(helper::gl::VisualParameters* vp)
{
    float vxmax, vymax, vzmax ;
    float vxmin, vymin, vzmin ;
    float txmax,tymax,tzmax;
    float txmin,tymin,tzmin;

    const Vec<2, unsigned int> &screenPosition = *p_screenPosition.beginEdit();
    const Vec<2, unsigned int> &screenSize = *p_screenSize.beginEdit();

    txmin = tymin = tzmin = 0.0;
    vxmin = vymin = vzmin = -1.0;
    vxmax = vymax = vzmax = txmax = tymax = tzmax = 1.0;

    glViewport(screenPosition[0],screenPosition[1],screenSize[0],screenSize[1]);

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

    glViewport(0,0,vp->viewport[2],vp->viewport[3]);
}

}

}

}
