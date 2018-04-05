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
/******************************************************************************
* Contributors:                                                               *
*    - damien.marchal@univ-lille1.fr Copyright (C) CNRS                       *
*******************************************************************************/

#include <PSL/config.h>

#include <PSL/components/PSLVersion.h>
#include <PSL/components/Import.h>
#include <PSL/components/TestResult.h>
#include <PSL/components/Template.h>
#include <PSL/SceneLoaderPSL.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <SofaPython/PythonEnvironment.h>
using sofa::simulation::PythonEnvironment ;

#include <SofaPython/PythonFactory.h>
using sofa::PythonFactory ;

extern "C" {
    SOFA_PSL_API void initExternalModule();
    SOFA_PSL_API const char* getModuleName();
    SOFA_PSL_API const char* getModuleVersion();
    SOFA_PSL_API const char* getModuleLicense();
    SOFA_PSL_API const char* getModuleDescription();
    SOFA_PSL_API const char* getModuleComponentList();
}

void initExternalModule()
{

    static bool first = true;
    if (first)
    {
        /// There is dependency with SofaPython
        PluginManager::getInstance().loadPlugin("SofaPython") ;
        first = false;
    }

    PythonEnvironment::gil lock();

    // Add the python classes in the Python Factory
    SP_ADD_CLASS_IN_FACTORY(Template,sofa::component::Template)
}

const char* getModuleName()
{
    return "PSL";
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
    return "This plugin contains a set of function to assist in the making of scenes by providing"
           " prefabs and other tools.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "";
}

SP_DECLARE_CLASS_TYPE(Template)
SOFA_LINK_CLASS(Import)
SOFA_LINK_CLASS(Python)
SOFA_LINK_CLASS(TestResult)
SOFA_LINK_CLASS(PSLVersion)

/// Use the SOFA_LINK_CLASS macro for each class, to enable linking on all platforms

/// register the loader in the factory
const sofa::simulation::SceneLoader* loaderPSL = sofa::simulation::SceneLoaderFactory::getInstance()->addEntry(new sofa::simulation::SceneLoaderPSL());
