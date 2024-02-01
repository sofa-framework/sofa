/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/statecontainer/init.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/statecontainer/MappedObject.h>
#include <sofa/component/statecontainer/MechanicalObject.h>

namespace sofa::component::statecontainer
{
    
extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleComponentList();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

void init()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    static bool registered = false;

    if (!registered)
    {
        using namespace sofa::defaulttype;
        
        // Registration with explicit commit
        // MappedObject
        core::RegisterObject("Mapped state vectors")
            .add< MappedObject<Vec1Types> >()
            .add< MappedObject<Vec3Types> >(true) // default template
            .add< MappedObject<Vec2Types> >()
            .add< MappedObject<Vec6Types> >()
            .add< MappedObject<Rigid3Types> >()
            .add< MappedObject<Rigid2Types> >()
            .commit(factory);

        // Registration with RAII-style commit
        // MechanicalObject
        core::RegisterObject("mechanical state vectors", factory)
            .add< MechanicalObject<Vec3Types> >(true) // default template
            .add< MechanicalObject<Vec2Types> >()
            .add< MechanicalObject<Vec1Types> >()
            .add< MechanicalObject<Vec6Types> >()
            .add< MechanicalObject<Rigid3Types> >()
            .add< MechanicalObject<Rigid2Types> >();

        registered = true;
    }
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = core::ObjectFactory::getInstance()->listClassesFromTarget(MODULE_NAME);
    return classes.c_str();
}
} // namespace sofa::component::statecontainer
