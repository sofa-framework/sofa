/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <TestPlugin/TestPlugin.h>

extern "C" {

static int counter = 0;

SOFA_TESTPLUGIN_API void initExternalModule()
{
    static bool first = true;

    if (first)
    {
        first = false;
    }
    counter++;
}

SOFA_TESTPLUGIN_API const char* getModuleName()
{
    return "TestPlugin";
}

SOFA_TESTPLUGIN_API const char* getModuleVersion()
{
    return "0.7";
}

SOFA_TESTPLUGIN_API const char* getModuleLicense()
{
    return "LicenceTest";
}

SOFA_TESTPLUGIN_API const char* getModuleDescription()
{
    return "Description of the Test Plugin";
}

SOFA_TESTPLUGIN_API const char* getModuleComponentList()
{
    return "ComponentA, ComponentB";
}

} // extern "C"


SOFA_LINK_CLASS(ComponentA)
SOFA_LINK_CLASS(ComponentB)
