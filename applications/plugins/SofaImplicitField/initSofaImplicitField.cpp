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
#include <SofaImplicitField/config.h>
#include <SofaImplicitField/initSofaImplicitField.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

namespace sofa::component::geometry::_BottleField_
{
    extern void registerBottleField(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::geometry::_sphericalfield_
{
    extern void registerSphericalField(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::geometry::_StarShapedField_
{
    extern void registerStarShapedField(sofa::core::ObjectFactory* factory);
}
namespace sofaimplicitfield::mapping
{
    extern void registerImplicitSurfaceMapping(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::container
{
    extern void registerInterpolatedImplicitSurface(sofa::core::ObjectFactory* factory);
}
namespace sofa::component::geometry::_discretegrid_
{
    extern void registerDiscreteGridField(sofa::core::ObjectFactory* factory);
}
namespace sofaimplicitfield::component::engine
{
extern void registerFieldToSurfaceMesh(sofa::core::ObjectFactory* factory);
}

namespace sofaimplicitfield
{

extern "C" {
    SOFA_SOFAIMPLICITFIELD_API void initExternalModule();
    SOFA_SOFAIMPLICITFIELD_API const char* getModuleName();
    SOFA_SOFAIMPLICITFIELD_API const char* getModuleVersion();
    SOFA_SOFAIMPLICITFIELD_API const char* getModuleLicense();
    SOFA_SOFAIMPLICITFIELD_API const char* getModuleDescription();
    SOFA_SOFAIMPLICITFIELD_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        first = false;
    }
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "ImplicitField describe shapes of objects using implicit equation.  \n"
           "In general of function of a n-dimentional space f(X) returns a scalar value  \n"
           "The surface is then defined as f(x) = aConstant.";
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    sofa::component::geometry::_BottleField_::registerBottleField(factory);
    sofa::component::geometry::_sphericalfield_::registerSphericalField(factory);
    sofa::component::geometry::_StarShapedField_::registerStarShapedField(factory);
    sofaimplicitfield::mapping::registerImplicitSurfaceMapping(factory);
    sofa::component::container::registerInterpolatedImplicitSurface(factory);
    sofa::component::geometry::_discretegrid_::registerDiscreteGridField(factory);
    sofaimplicitfield::component::engine::registerFieldToSurfaceMesh(factory);
}

} /// sofaimplicitfield
