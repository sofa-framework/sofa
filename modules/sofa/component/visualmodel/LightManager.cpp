/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
