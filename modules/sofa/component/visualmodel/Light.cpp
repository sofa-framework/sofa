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
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

SOFA_DECL_CLASS(Light)
SOFA_DECL_CLASS(LightTable)
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

LightTable* LightTable::instance = NULL;

Light::Light():
    color(initData(&color, (Vector3) Vector3(1,0,0), "color", "Set the color of the light"))
{

}

Light::~Light()
{
    if (lightID != GL_LIGHT0)
        glDisable(lightID);
    else
    {
        //glEnable(GL_LIGHT0);
        GLfloat amb[4] = { 0.0, 0.0, 0.0, 1.0 };
        GLfloat diff[4] = { 1.0, 1.0, 1.0, 1.0 };
        GLfloat spec[4] = { 0.0, 0.0, 0.0, 1.0 };
// 		GLfloat c[4] = { 1.0, 1.0, 1.0, 1.0 };
        GLfloat pos[4] = { 0.0, 0.0, 1.0, 0.0 };
        glLightfv(lightID, GL_AMBIENT, amb);
        glLightfv(lightID, GL_DIFFUSE, diff);
        glLightfv(lightID, GL_SPECULAR, spec);
        glLightfv(lightID, GL_POSITION, pos );

    }
    LightTable::getInstance()->removeLightID(name.getValue());
}

void Light::init()
{
    lightID =  LightTable::getInstance()->getAvailableLightID(name.getValue());

}

void Light::initTextures()
{
    glEnable(GL_LIGHTING);
    glEnable(lightID);

    GLfloat c[4] = { (GLfloat) color.getValue()[0], (GLfloat)color.getValue()[1], (GLfloat)color.getValue()[2], 1.0 };
    glLightfv(lightID, GL_AMBIENT, c);
    glLightfv(lightID, GL_DIFFUSE, c);
    glLightfv(lightID, GL_SPECULAR, c);
}

void Light::reinit()
{

    initTextures();

}

void Light::draw()
{


}

DirectionalLight::DirectionalLight():
    direction(initData(&direction, (Vector3) Vector3(0,0,-1), "direction", "Set the direction of the light"))
{

}

DirectionalLight::~DirectionalLight()
{

}

void DirectionalLight::initTextures()
{
    Light::initTextures();

}

void DirectionalLight::reinit()
{
    initTextures();
}

void DirectionalLight::draw()
{
    Light::draw();

    GLfloat dir[4];
    dir[0]=(GLfloat)(direction.getValue()[0]);
    dir[1]=(GLfloat)(direction.getValue()[1]);
    dir[2]=(GLfloat)(direction.getValue()[2]);
    dir[3]=0.0; // directional

    glLightfv(lightID, GL_POSITION, dir);
}

PositionalLight::PositionalLight():
    position(initData(&position, (Vector3) Vector3(0,0,-1), "position", "Set the position of the light")),
    attenuation(initData(&attenuation, (float) 0.0, "attenuation", "Set the attenuation of the light"))
{

}

PositionalLight::~PositionalLight()
{

}

void PositionalLight::initTextures()
{
    Light::initTextures();

}

void PositionalLight::reinit()
{
    initTextures();

}

void PositionalLight::draw()
{
    Light::draw();

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
    cutoff(initData(&cutoff, (float) 30.0, "cutoff", "Set the angle (cutoff) of the spot"))
{

}

SpotLight::~SpotLight()
{

}

void SpotLight::initTextures()
{
    PositionalLight::initTextures();

}

void SpotLight::reinit()
{
    initTextures();

}

void SpotLight::draw()
{
    PositionalLight::draw();

    GLfloat dir[]= {(GLfloat)(direction.getValue()[0]), (GLfloat)(direction.getValue()[1]), (GLfloat)(direction.getValue()[2])};
    glLightf(lightID, GL_SPOT_CUTOFF, cutoff.getValue());
    glLightfv(lightID, GL_SPOT_DIRECTION, dir);
    glLightf(lightID, GL_SPOT_EXPONENT, 20.0);
}


} //namespace component

} //namespace sofa
