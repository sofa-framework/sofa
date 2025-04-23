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

namespace sofaimplicitfield
{

extern "C" {
SOFA_SOFAIMPLICITFIELD_API void initExternalModule();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleName();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleVersion();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleLicense();
SOFA_SOFAIMPLICITFIELD_API const char* getModuleDescription();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
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

}

