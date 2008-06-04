//
// C++ Implementation: Light
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <sofa/component/visualmodel/Light.h>
#include <sofa/component/visualmodel/LightManager.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(Light)

SOFA_DECL_CLASS(DirectionalLight)
//Register DirectionalLight in the Object Factory
int DirectionalLightClass = core::RegisterObject("Directional Light")
        .add< DirectionalLight >()
        ;

SOFA_DECL_CLASS(PositionalLight)
//Register PositionalLight in the Object Factory
int PositionalLightClass = core::RegisterObject("Positional Light")
        .add< PositionalLight >()
        ;

SOFA_DECL_CLASS(SpotLight)
//Register SpotLight in the Object Factory
int SpotLightClass = core::RegisterObject("Spot Light")
        .add< SpotLight >()
        ;


Light::Light():
    color(initData(&color, (Vector3) Vector3(1,1,1), "color", "Set the color of the light"))
{
    lightID = GL_LIGHT0;
}

Light::~Light()
{
}

void Light::setID(const GLint& id)
{
    lightID = id;
}

void Light::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    LightManager* lm = context->core::objectmodel::BaseContext::get<LightManager>();

    lm->putLight(this);

}

void Light::initVisual()
{

    glLightf(lightID, GL_SPOT_CUTOFF, 180.0);
    GLfloat c[4] = { (GLfloat) color.getValue()[0], (GLfloat)color.getValue()[1], (GLfloat)color.getValue()[2], 1.0 };
    glLightfv(lightID, GL_AMBIENT, c);
    glLightfv(lightID, GL_DIFFUSE, c);
    glLightfv(lightID, GL_SPECULAR, c);
    glLightf(lightID, GL_LINEAR_ATTENUATION, 0.0);

}

void Light::reinit()
{

    initVisual();

}

void Light::drawLight()
{

}

DirectionalLight::DirectionalLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light"))
{

}

DirectionalLight::~DirectionalLight()
{

}

void DirectionalLight::initVisual()
{
    Light::initVisual();

}

void DirectionalLight::reinit()
{
    initVisual();
}

void DirectionalLight::drawLight()
{
    Light::drawLight();
    GLfloat dir[4];

    dir[0]=(GLfloat)(direction.getValue()[0]);
    dir[1]=(GLfloat)(direction.getValue()[1]);
    dir[2]=(GLfloat)(direction.getValue()[2]);
    dir[3]=0.0; // directional

    glLightfv(lightID, GL_POSITION, dir);
}

PositionalLight::PositionalLight():
    position(initData(&position, (Vector3) Vector3(-0.7,0.3,0.0), "position", "Set the position of the light")),
    attenuation(initData(&attenuation, (float) 0.0, "attenuation", "Set the attenuation of the light"))
{

}

PositionalLight::~PositionalLight()
{

}

void PositionalLight::initVisual()
{
    Light::initVisual();

}

void PositionalLight::reinit()
{
    initVisual();

}

void PositionalLight::drawLight()
{
    Light::drawLight();

    GLfloat pos[4];
    pos[0]=(GLfloat)(position.getValue()[0]);
    pos[1]=(GLfloat)(position.getValue()[1]);
    pos[2]=(GLfloat)(position.getValue()[2]);
    pos[3]=1.0; // positional
    glLightfv(lightID, GL_POSITION, pos);

    glLightf(lightID, GL_LINEAR_ATTENUATION, attenuation.getValue());

}


SpotLight::SpotLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light")),
    cutoff(initData(&cutoff, (float) 30.0, "cutoff", "Set the angle (cutoff) of the spot")),
    exponent(initData(&exponent, (float) 20.0, "exponent", "Set the exponent of the spot"))
{

}

SpotLight::~SpotLight()
{

}

void SpotLight::initVisual()
{
    PositionalLight::initVisual();

}

void SpotLight::reinit()
{
    initVisual();

}

void SpotLight::drawLight()
{
    PositionalLight::drawLight();

    GLfloat dir[]= {(GLfloat)(direction.getValue()[0]), (GLfloat)(direction.getValue()[1]), (GLfloat)(direction.getValue()[2])};
    glLightf(lightID, GL_SPOT_CUTOFF, cutoff.getValue());
    glLightfv(lightID, GL_SPOT_DIRECTION, dir);
    glLightf(lightID, GL_SPOT_EXPONENT, exponent.getValue());

}

}

} //namespace component

} //namespace sofa
