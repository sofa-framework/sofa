/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaImplicitField/config.h>

#include "initSofaImplicitField.h"
#include "components/geometry/ScalarField.h"
#include "components/geometry/SphericalField.h"
#include "components/geometry/DiscreteGridField.h"

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#ifdef SOFA_HAVE_SOFAPYTHON
#define DO_SOFAPYTHON_FEATURES true
#else
#define DO_SOFAPYTHON_FEATURES false
#endif
namespace sofa
{

namespace component
{
extern "C" {
SOFA_SOFAIMPLICITFIELD_API void initExternalModule();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleName();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleVersion();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleLicense();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleDescription();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }

    if(DO_SOFAPYTHON_FEATURES)
        PluginManager::getInstance().loadPlugin("SofaPython") ;
}

const char* getModuleName()
{
    return "SofaImplicitField";
}

const char* getModuleVersion()
{
    return "1.0";
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

const char* getModuleComponentList()
{
    return "SphereSurface ImplicitSurfaceMapping InterpolatedImplicitSurface "
           "SphericalField DiscreteGridField";
}

SOFA_LINK_CLASS(SphericalField)
SOFA_LINK_CLASS(DiscreteGridField)


} /// component

} /// sofa

