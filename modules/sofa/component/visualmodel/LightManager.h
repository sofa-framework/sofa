//
// C++ Interface: LightManager
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_LIGHTMANAGER_H
#define SOFA_COMPONENT_LIGHTMANAGER_H

#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/component/visualmodel/Light.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class LightManager : public core::VisualModel
{
private:
    static const unsigned int MAX_NUMBER_OF_LIGHTS = GL_MAX_LIGHTS;
    std::vector<Light*> lights;
    Light* defaultLight;

public:
    LightManager();
    virtual ~LightManager();

    void init();
    void reinit();
    void initVisual();
    void update() { };
    void draw();
    void putLight(Light* light);
    void putLights(std::vector<Light*> lights);
    void clear();
};

}//namespace visualmodel

}//namespace component

}//namespace sofa

#endif //SOFA_COMPONENT_LIGHT_MANAGER_H
