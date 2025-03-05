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
#include <TestPluginC/TestPluginC.h>

#include <sofa/core/ObjectFactory.h>

namespace testpluginc
{

extern void registerComponentD(sofa::core::ObjectFactory* factory);

extern "C" SOFA_EXPORT_DYNAMIC_LIBRARY  void initExternalModule()
{
    init();
}

extern "C" SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName()
{
    return "TestPluginC";
}

extern "C" SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerComponentD(factory);
}

SOFA_EXPORT_DYNAMIC_LIBRARY void init()
{
    static bool first = true;

    if (first)
    {
        first = false;
    }
}

}
