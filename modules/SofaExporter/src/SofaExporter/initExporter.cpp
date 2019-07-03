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
#include <SofaExporter/config.h>

#include <sofa/core/ObjectFactory.h>

#if SOFA_HAVE_SOFAPYTHON
#include <SofaPython/PythonEnvironment.h>
#include <SofaPython/PythonFactory.h>

using sofa::simulation::PythonEnvironment ;
using sofa::PythonFactory ;

#include <SofaExporter/bindings/Binding_OBJExporter.h>
#include <SofaExporter/bindings/Binding_STLExporter.h>

#endif // SOFA_HAVE_SOFAPYTHON


using sofa::core::ObjectFactory;

namespace sofa
{

namespace component
{

extern "C" {
SOFA_SOFAEXPORTER_API void initExternalModule();
SOFA_SOFAEXPORTER_API const char* getModuleName();
SOFA_SOFAEXPORTER_API const char* getModuleVersion();
SOFA_SOFAEXPORTER_API const char* getModuleLicense();
SOFA_SOFAEXPORTER_API const char* getModuleDescription();
SOFA_SOFAEXPORTER_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
#ifdef SOFA_HAVE_SOFAPYTHON
        {
            SP_ADD_CLASS_IN_FACTORY(OBJExporter,sofa::component::misc::OBJExporter)
            SP_ADD_CLASS_IN_FACTORY(STLExporter,sofa::component::misc::STLExporter)
        }
#endif
    }
}

const char* getModuleName()
{
    return "SofaExporter";
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
    return "This plugin contains some exporter to save simulation scenes to various formats. "
            "Supported format are: Sofa internal state format, VTK, STL, Mesh, Blender.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace component

} // namespace sofa
