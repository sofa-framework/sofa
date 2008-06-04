//
// C++ Implementation: LightManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/component/visualmodel/LightManager.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(LightManager)
//Register LightManager in the Object Factory
int LightManagerClass = core::RegisterObject("LightManager")
        .add< LightManager >()
        ;

LightManager::LightManager()
{

}

LightManager::~LightManager()
{

}

void LightManager::init()
{

}

void LightManager::putLight(Light* light)
{
    if (lights.size() >= MAX_NUMBER_OF_LIGHTS)
    {
        std::cerr << "The maximum of lights permitted ( "<< MAX_NUMBER_OF_LIGHTS << " ) has been reached." << std::endl;
        return ;
    }

    light->setID(GL_LIGHT0+lights.size());
    lights.push_back(light) ;
}

void LightManager::putLights(std::vector<Light*> lights)
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
        putLight(*itl);
}

void LightManager::draw()
{
    unsigned int id = 0;
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        glEnable(GL_LIGHT0+id);
        (*itl)->drawLight();
        id++;
    }
}

void LightManager::initVisual()
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        (*itl)->initVisual();
    }

}


void LightManager::clear()
{
    for (unsigned int i=0 ; i<MAX_NUMBER_OF_LIGHTS ; i++)
        glDisable(GL_LIGHT0+i);
    lights.clear();
}

void LightManager::reinit()
{
    for (std::vector<Light*>::iterator itl = lights.begin(); itl != lights.end() ; itl++)
    {
        (*itl)->reinit();
    }
}

}//namespace visualmodel

}//namespace component

}//namespace sofa
