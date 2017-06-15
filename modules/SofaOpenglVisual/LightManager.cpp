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
#include <SofaOpenglVisual/LightManager.h>
using sofa::component::visualmodel::OglShadowShader;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams ;

#include <sofa/simulation/VisualVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#ifdef SOFA_HAVE_GLEW
#include <SofaOpenglVisual/OglTexture.h>
using sofa::component::visualmodel::OglTexture ;
#endif // SOFA_HAVE_GLEW

using sofa::core::objectmodel::BaseContext ;
using sofa::core::RegisterObject ;

using sofa::defaulttype::Mat ;

using sofa::helper::types::RGBAColor ;

namespace sofa
{

namespace component
{

namespace visualmodel
{


//TODO(dmarchal): There is a large amount of #ifdef SOFA_HAVE_GLEW why ? Too much #ifdef is
//a sign it is time to refactor the code.

SOFA_DECL_CLASS(LightManager)

//Register LightManager in the Object Factory
int LightManagerClass = RegisterObject
        ("Manage a set of lights that can cast hard and soft shadows.Soft Shadows is done using Variance Shadow Mapping "
         "(http://developer.download.nvidia.com/SDK/10.5/direct3d/Source/VarianceShadowMapping/Doc/VarianceShadowMapping.pdf)")
        .add< LightManager >()
        ;

LightManager::LightManager()
    : d_shadowsEnabled(initData(&d_shadowsEnabled, (bool) false, "shadows", "Enable Shadow in the scene. (default=0)"))
    , d_softShadowsEnabled(initData(&d_softShadowsEnabled, (bool) false, "softShadows", "If Shadows enabled, Enable Variance Soft Shadow in the scene. (default=0)"))
    , d_ambient(initData(&d_ambient, RGBAColor::black(), "ambient", "Ambient lights contribution (Vec4f)(default=[0.0f,0.0f,0.0f,0.0f])"))
    , d_drawIsEnabled(initData(&d_drawIsEnabled, false, "debugDraw", "enable/disable drawing of lights shadow textures. (default=false)"))
{
    //listen by default, in order to get the keys to activate shadows
    if(!f_listening.isSet())
        f_listening.setValue(true);
}

LightManager::~LightManager()
{
    //restoreDefaultLight();
}

void LightManager::init()
{
    BaseContext* context = this->getContext();
#ifdef SOFA_HAVE_GLEW
    context->get<OglShadowShader, helper::vector<OglShadowShader::SPtr> >(&m_shadowShaders, BaseContext::SearchRoot);

    if (m_shadowShaders.empty() && d_shadowsEnabled.getValue())
    {
        msg_warning(this) << "No OglShadowShaders found ; shadow will be disabled." ;
        d_shadowsEnabled.setValue(false);
    }

    for(unsigned int i=0 ; i<m_shadowShaders.size() ; ++i)
    {
        m_shadowShaders[i]->initShaders((unsigned int)m_lights.size(), d_softShadowsEnabled.getValue());
        m_shadowShaders[i]->setCurrentIndex(d_shadowsEnabled.getValue() ? 1 : 0);
    }
#endif
    m_lightModelViewMatrix.resize(m_lights.size());
}

void LightManager::bwdInit()
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<m_shadowShaders.size() ; ++i)
    {
        m_shadowShaders[i]->setCurrentIndex(d_shadowsEnabled.getValue() ? 1 : 0);
    }
#endif
}

void LightManager::initVisual()
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<m_shadowShaders.size() ; ++i)
        m_shadowShaders[i]->initVisual();

    ///TODO: keep trace of all active textures at the same time, with a static factory
    ///or something like that to avoid conflics with color texture declared in the scene file.
    helper::vector<OglTexture::SPtr> sceneTextures;
    this->getContext()->get<OglTexture, helper::vector<OglTexture::SPtr> >(&sceneTextures, BaseContext::SearchRoot);

    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    std::vector<bool> availableUnitTextures;
    availableUnitTextures.resize(maxTextureUnits);
    std::fill(availableUnitTextures.begin(), availableUnitTextures.end(), true);
    for(unsigned int i=0 ; i<sceneTextures.size() ; i++)
    {
        availableUnitTextures[sceneTextures[i]->getTextureUnit()] = false;
    }

    for (std::vector<Light::SPtr>::iterator itl = m_lights.begin(); itl != m_lights.end() ; ++itl)
    {
        (*itl)->initVisual();
        unsigned short shadowTextureUnit = (*itl)->getShadowTextureUnit();

        /// if given unit is available and correct
        if(shadowTextureUnit < maxTextureUnits &&
                availableUnitTextures[shadowTextureUnit] == true)
        {
            availableUnitTextures[shadowTextureUnit] = false;
        }
        /// otherwise search the first one available
        else
        {
            bool found = false;
            for(unsigned short i=0 ; i < availableUnitTextures.size() && !found; i++)
            {
                found = availableUnitTextures[i];
                if(found)
                {
                    (*itl)->setShadowTextureUnit(i);
                    availableUnitTextures[i] = false;
                }

            }

        }
    }

#endif // SOFA_HAVE_GLEW
}

void LightManager::putLight(Light::SPtr light)
{
    if (m_lights.size() >= MAX_NUMBER_OF_LIGHTS)
    {
        msg_error(this) << "The maximum of lights permitted ( "<< MAX_NUMBER_OF_LIGHTS << " ) has been reached." ;
        return ;
    }

    light->setID((GLint)m_lights.size());
    m_lights.push_back(light) ;
}

void LightManager::putLights(std::vector<Light::SPtr> lights)
{
    for (std::vector<Light::SPtr>::iterator itl = lights.begin(); itl != lights.end() ; ++itl)
        putLight(*itl);
}

void LightManager::makeShadowMatrix(unsigned int i)
{
    const GLfloat* lp = m_lights[i]->getOpenGLProjectionMatrix();
    const GLfloat* lmv = m_lights[i]->getOpenGLModelViewMatrix();

    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glTranslatef(0.5f, 0.5f, 0.5f +( -0.006f) );
    glScalef(0.5f, 0.5f, 0.5f);

    // now multiply by the matrices we have retrieved before
    glMultMatrixf(lp);
    glMultMatrixf(lmv);
    Mat<4,4,float> model2;
    glGetFloatv(GL_MODELVIEW_MATRIX,model2.ptr());
    model2.invert(model2);

    glMultMatrixf(model2.ptr());
    if (m_lightModelViewMatrix.size() > 0)
    {
        m_lightModelViewMatrix[i] = lmv;
        m_lightProjectionMatrix[i] = lp;
    }
    else
    {
        m_lightModelViewMatrix.resize(m_lights.size());
        m_lightProjectionMatrix.resize(m_lights.size());
        m_lightModelViewMatrix[i] = lmv;
        m_lightProjectionMatrix[i] = lp;
    }

    glMatrixMode(GL_MODELVIEW);
}

void LightManager::fwdDraw(core::visual::VisualParams* vp)
{

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, d_ambient.getValue().data());
    unsigned int id = 0;
    for (std::vector<Light::SPtr>::iterator itl = m_lights.begin(); itl != m_lights.end() ; ++itl)
    {
        glEnable(GL_LIGHT0+id);
        (*itl)->drawLight();
        ++id;
    }

#ifdef SOFA_HAVE_GLEW
    const core::visual::VisualParams::Pass pass = vp->pass();
    GLint lightFlags[MAX_NUMBER_OF_LIGHTS];
    GLint lightTypes[MAX_NUMBER_OF_LIGHTS];
    GLint shadowTextureID[MAX_NUMBER_OF_LIGHTS];
    GLfloat zNears[MAX_NUMBER_OF_LIGHTS];
    GLfloat zFars[MAX_NUMBER_OF_LIGHTS];
    GLfloat lightDirs[MAX_NUMBER_OF_LIGHTS * 3];
    GLfloat lightProjectionMatrices[MAX_NUMBER_OF_LIGHTS * 16];
    GLfloat lightModelViewMatrices[MAX_NUMBER_OF_LIGHTS * 16];
    GLfloat shadowFactors[MAX_NUMBER_OF_LIGHTS];
    GLfloat vsmLightBleedings[MAX_NUMBER_OF_LIGHTS];
    GLfloat vsmMinVariances[MAX_NUMBER_OF_LIGHTS];

    if(pass != core::visual::VisualParams::Shadow)
    {
        if (!m_shadowShaders.empty())
        {
            glEnable(GL_LIGHTING);
            for (unsigned int i=0 ; i < m_lights.size() ; ++i)
            {
                unsigned short shadowTextureUnit = m_lights[i]->getShadowTextureUnit();
                glActiveTexture(GL_TEXTURE0+shadowTextureUnit);
                glEnable(GL_TEXTURE_2D);

                if (d_softShadowsEnabled.getValue())
                {
                    glBindTexture(GL_TEXTURE_2D, m_lights[i]->getColorTexture());
                    vsmLightBleedings[i] = m_lights[i]->getVSMLightBleeding();
                    vsmMinVariances[i] = m_lights[i]->getVSMMinVariance();
                }
                else
                    glBindTexture(GL_TEXTURE_2D, m_lights[i]->getColorTexture());

                lightFlags[i] = 1;
                shadowTextureID[i] = 0;

                zNears[i] = (GLfloat) m_lights[i]->getZNear();
                zFars[i] = (GLfloat) m_lights[i]->getZFar();

                for (unsigned int j = 0; j < 3; j++)
                    lightDirs[i*3 + j] = (m_lights[i]->getDirection())[j];
                lightTypes[i] = m_lights[i]->getLightType();


                const GLfloat* tmpProj = m_lights[i]->getOpenGLProjectionMatrix();
                const GLfloat* tmpMv = m_lights[i]->getOpenGLModelViewMatrix();
                for (unsigned int j = 0; j < 16; j++)
                {
                    lightProjectionMatrices[i * 16 + j] = tmpProj[j];
                    lightModelViewMatrices[i * 16 + j] = tmpMv[j];
                }

                if (d_shadowsEnabled.getValue() && m_lights[i]->d_shadowsEnabled.getValue())
                {
                    lightFlags[i] = 2;
                    shadowTextureID[i] = shadowTextureUnit;
                }
                shadowFactors[i] = m_lights[i]->getShadowFactor();

                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
                glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB);
                glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

                makeShadowMatrix(i);
            }

            for (unsigned int i = (unsigned int)m_lights.size() ; i< MAX_NUMBER_OF_LIGHTS ; i++)
            {
                lightFlags[i] = 0;
                shadowTextureID[i] = 0;
            }

            for(unsigned int i=0 ; i<m_shadowShaders.size() ; ++i)
            {
                m_shadowShaders[i]->setIntVector(m_shadowShaders[i]->getCurrentIndex() , "u_lightFlags" , MAX_NUMBER_OF_LIGHTS, lightFlags);
                m_shadowShaders[i]->setIntVector(m_shadowShaders[i]->getCurrentIndex(), "u_lightTypes", MAX_NUMBER_OF_LIGHTS, lightTypes);
                m_shadowShaders[i]->setIntVector(m_shadowShaders[i]->getCurrentIndex() , "u_shadowTextures" , MAX_NUMBER_OF_LIGHTS, shadowTextureID);
                m_shadowShaders[i]->setIntVector(m_shadowShaders[i]->getCurrentIndex() , "u_shadowTextureUnits" , MAX_NUMBER_OF_LIGHTS, shadowTextureID);
                m_shadowShaders[i]->setFloatVector(m_shadowShaders[i]->getCurrentIndex() , "u_zNears" , MAX_NUMBER_OF_LIGHTS, zNears);
                m_shadowShaders[i]->setFloatVector(m_shadowShaders[i]->getCurrentIndex() , "u_zFars" , MAX_NUMBER_OF_LIGHTS, zFars);
                m_shadowShaders[i]->setFloatVector3(m_shadowShaders[i]->getCurrentIndex(), "u_lightDirs", MAX_NUMBER_OF_LIGHTS, lightDirs);
                m_shadowShaders[i]->setMatrix4(m_shadowShaders[i]->getCurrentIndex(), "u_lightProjectionMatrices", MAX_NUMBER_OF_LIGHTS, false, lightProjectionMatrices);
                m_shadowShaders[i]->setMatrix4(m_shadowShaders[i]->getCurrentIndex(), "u_lightModelViewMatrices", MAX_NUMBER_OF_LIGHTS, false, lightModelViewMatrices);
                m_shadowShaders[i]->setFloatVector(m_shadowShaders[i]->getCurrentIndex(), "u_shadowFactors", MAX_NUMBER_OF_LIGHTS, shadowFactors);
            }

            if (d_softShadowsEnabled.getValue())
            {
                for (unsigned int i = 0; i < m_shadowShaders.size(); ++i)
                {
                    m_shadowShaders[i]->setFloatVector(m_shadowShaders[i]->getCurrentIndex(), "u_lightBleedings", MAX_NUMBER_OF_LIGHTS, vsmLightBleedings);
                    m_shadowShaders[i]->setFloatVector(m_shadowShaders[i]->getCurrentIndex(), "u_minVariances", MAX_NUMBER_OF_LIGHTS, vsmMinVariances);
                }
            }



        }
    }

    glActiveTexture(GL_TEXTURE0);
#endif
}

void LightManager::bwdDraw(core::visual::VisualParams* )
{
#ifdef SOFA_HAVE_GLEW
    for(unsigned int i=0 ; i<m_lights.size() ; ++i)
    {
        unsigned short shadowTextureUnit = m_lights[i]->getShadowTextureUnit();
        glActiveTexture(GL_TEXTURE0+shadowTextureUnit);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    glActiveTexture(GL_TEXTURE0);
#endif // SOFA_HAVE_GLEW

    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; ++i)
        glDisable(GL_LIGHT0+i);

    //reset Texture Matrix
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);


}

void LightManager::draw(const core::visual::VisualParams* )
{
    if(!d_drawIsEnabled.getValue())
        return ;

    //reset Texture Matrix
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    for(unsigned int i=0 ; i < m_lights.size() ; i++)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_lights[i]->getDepthTexture());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_NONE);
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_LUMINANCE);

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glBegin(GL_QUADS);
        glColor3f(1.0,0.0,0.0) ; glTexCoord2f(0,0); glVertex3f(0 + i*20, 20, -10);
        glColor3f(0.0,1.0,0.0) ; glTexCoord2f(1,0); glVertex3f(0 + i*20, 40, -10);
        glColor3f(0.0,0.0,1.0) ; glTexCoord2f(1,1); glVertex3f(20 + i*20, 40, -10);
        glColor3f(0.0,0.0,0.0) ; glTexCoord2f(0,1); glVertex3f(20 + i*20, 20, -10);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, m_lights[i]->getColorTexture());

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glBegin(GL_QUADS);
        glColor3f(1.0,0.0,0.0) ; glTexCoord2f(0,0); glVertex3f(40 + i*20, 20, -10);
        glColor3f(0.0,1.0,0.0) ; glTexCoord2f(1,0); glVertex3f(40 + i*20, 40, -10);
        glColor3f(0.0,0.0,1.0) ; glTexCoord2f(1,1); glVertex3f(60 + i*20, 40, -10);
        glColor3f(0.0,0.0,0.0) ; glTexCoord2f(0,1); glVertex3f(60 + i*20, 20, -10);
        glEnd();
    }

    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glEnable(GL_LIGHTING);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void LightManager::clear()
{
    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; ++i)
        glDisable(GL_LIGHT0+i);
    m_lights.clear();
}

void LightManager::reinit()
{
    for (std::vector<Light::SPtr>::iterator itl = m_lights.begin(); itl != m_lights.end() ; ++itl)
    {
        (*itl)->reinit();
    }
}

void LightManager::preDrawScene(VisualParams* vp)
{
#ifdef SOFA_HAVE_GLEW
    if(d_shadowsEnabled.getValue())
    {
        for (std::vector<Light::SPtr>::iterator itl = m_lights.begin(); itl != m_lights.end() ; ++itl)
        {
            (*itl)->preDrawShadow(vp);
            vp->pass() = core::visual::VisualParams::Shadow;
            simulation::VisualDrawVisitor vdv(vp);

            vdv.execute ( getContext() );

            (*itl)->postDrawShadow();
        }
        const core::visual::VisualParams::Viewport& viewport = vp->viewport();
        //restore viewport
        glViewport(viewport[0], viewport[1], viewport[2] , viewport[3]);
    }
#endif
}

bool LightManager::drawScene(VisualParams* /*vp*/)
{
    return false;
}

void LightManager::postDrawScene(VisualParams* vp)
{
    restoreDefaultLight(vp);
}

void LightManager::restoreDefaultLight(VisualParams* vp)
{
    //restore default light
    GLfloat	ambientLight[4];
    GLfloat	diffuseLight[4];
    GLfloat	specular[4];
    GLfloat	lightPosition[4];

    lightPosition[0] = -0.7f;
    lightPosition[1] = 0.3f;
    lightPosition[2] = 0.0f;
    lightPosition[3] = 1.0f;

    ambientLight[0] = 0.5f;
    ambientLight[1] = 0.5f;
    ambientLight[2] = 0.5f;
    ambientLight[3] = 1.0f;

    diffuseLight[0] = 0.9f;
    diffuseLight[1] = 0.9f;
    diffuseLight[2] = 0.9f;
    diffuseLight[3] = 1.0f;

    specular[0] = 1.0f;
    specular[1] = 1.0f;
    specular[2] = 1.0f;
    specular[3] = 1.0f;

    // Setup 'light 0'
    // It crashes here in batch mode on Mac... probably the lack of GL context ?
    if (vp->isSupported(core::visual::API_OpenGL))
    {
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
        glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180);

        glEnable(GL_LIGHT0);
    }
}

//TODO(dmarchal): Hard-coding keyboard behavior in a component is a bad idea as for several reasons:
// the scene can be executed without a keyboard ...there is no reason the component should have a "knowledge" of keyboard
// what will happens if multiple lighmanager are in the same scene ...
// what will hapen if other component use the same key...
// The correct implementation consist in separatng the event code into a different class & component in
// the SofaInteracton module.
void LightManager::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'l':
        case 'L':
#ifdef SOFA_HAVE_GLEW
            if (!m_shadowShaders.empty())
            {
                bool b = d_shadowsEnabled.getValue();
                d_shadowsEnabled.setValue(!b);
                if (!m_shadowShaders.empty())
                {
                    for (unsigned int i=0 ; i < m_shadowShaders.size() ; ++i)
                    {
                        m_shadowShaders[i]->setCurrentIndex(d_shadowsEnabled.getValue() ? 1 : 0);
                        m_shadowShaders[i]->updateVisual();
                    }
                    for (std::vector<Light::SPtr>::iterator itl = m_lights.begin(); itl != m_lights.end() ; ++itl)
                    {
                        (*itl)->updateVisual();
                    }
                    this->updateVisual();
                }

                sout << "Shadows : "<<(d_shadowsEnabled.getValue()?"ENABLED":"DISABLED")<<sendl;
            }
#endif
            break;
        }
    }

}

}//namespace visualmodel

}//namespace component

}//namespace sofa
