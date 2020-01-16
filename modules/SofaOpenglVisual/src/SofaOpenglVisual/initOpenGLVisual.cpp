/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <string>
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject;

#include <sofa/helper/system/config.h>
#include <SofaOpenglVisual/initOpenGLVisual.h>

#include <SofaOpenglVisual/OglModel.h>

namespace sofa
{

namespace component
{

extern "C" {
    SOFA_OPENGL_VISUAL_API void initExternalModule();
    SOFA_OPENGL_VISUAL_API const char* getModuleName();
    SOFA_OPENGL_VISUAL_API const char* getModuleVersion();
    SOFA_OPENGL_VISUAL_API const char* getModuleLicense();
    SOFA_OPENGL_VISUAL_API const char* getModuleDescription();
    SOFA_OPENGL_VISUAL_API const char* getModuleComponentList();
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
    return "SofaOpenglVisual";
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
    return "Visual object rendered using OpenGL 1.x/2.x.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace component

} // namespace sofa
