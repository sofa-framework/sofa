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
#include "SceneLoaderPY.h"
#include <SofaPython/config.h>
#include "PythonEnvironment.h"

#include <SofaSimulationCommon/init.h>
#ifdef SOFA_HAVE_DAG
#include <SofaSimulationGraph/init.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#endif
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>
#include <sofa/gui/Main.h>
#include "Binding_SofaModule.h"
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/BackTrace.h>


extern "C" {

SOFA_SOFAPYTHON_API void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        sofa::simulation::PythonEnvironment::Init();
        first = false;
    }
}

SOFA_SOFAPYTHON_API const char* getModuleName()
{
    return "SofaPython";
}

SOFA_SOFAPYTHON_API const char* getModuleVersion()
{
    return SOFAPYTHON_VERSION_STR;
}

SOFA_SOFAPYTHON_API const char* getModuleLicense()
{
    return "LGPL";
}

SOFA_SOFAPYTHON_API const char* getModuleDescription()
{
    return "Python Environment and modules for scripting in Sofa";
}

SOFA_SOFAPYTHON_API const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "PythonScriptController";
}

}

/// Use the SOFA_LINK_CLASS macro for each class, to enable linking on all platforms
/// register the loader in the factory
const sofa::simulation::SceneLoader* loaderPY = sofa::simulation::SceneLoaderFactory::getInstance()->addEntry(new sofa::simulation::SceneLoaderPY());

// create an entry point for dynamic loading of the plugin inside a Python program

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INIT_LIBRARY_A(NAME) extern "C" void init ##NAME (void)
#define INIT_LIBRARY(NAME) INIT_LIBRARY_A(NAME)

INIT_LIBRARY(LIBRARY_NAME)
{
    sofa::helper::BackTrace::autodump();

    sofa::helper::system::Plugin p;
    p.permanent = true;
    p.initExternalModule.func      = &initExternalModule;
    p.getModuleVersion.func        = &getModuleVersion;
    p.getModuleComponentList.func  = &getModuleComponentList;
    p.getModuleName.func           = &getModuleName;
    p.getModuleDescription.func    = &getModuleDescription;
    p.getModuleLicense.func        = &getModuleLicense;
    std::string pluginPath = TOSTRING(LIBRARY_NAME) + std::string(".so");
    sofa::helper::system::PluginManager::getInstance().addPlugin(pluginPath, p);


    PyObject * libraryModule = Py_InitModule(TOSTRING(LIBRARY_NAME), NULL);

    sofa::simulation::tree::init();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::init();
#endif
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();
    sofa::gui::initMain();


    PyObject * sofaModule = Py_InitModule("Sofa", SofaModuleMethods);
    bindSofaPythonModule(sofaModule);
    PyModule_AddObject(libraryModule, "Sofa", sofaModule);

    sofa::simulation::SceneLoaderFactory::getInstance()->removeEntry(loaderPY);
}
