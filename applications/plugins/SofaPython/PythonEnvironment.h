/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef PYTHONENVIRONMENT_H
#define PYTHONENVIRONMENT_H

#include "PythonCommon.h"
#include "PythonMacros.h"

#include "Binding.h"
#include <SofaPython/config.h>
#include <sofa/simulation/SceneLoaderFactory.h>
#include <vector>
#include <string>

namespace sofa
{

namespace simulation
{

class SOFA_SOFAPYTHON_API PythonEnvironment
{
public:
    static void Init();
    static void Release();

    /// Add a path to sys.path, the list of search path for Python modules.
    static void addPythonModulePath(const std::string& path);

    /// Add each line of a file to sys.path
    static void addPythonModulePathsFromConfigFile(const std::string& path);

    /// Add all the directories matching <pluginsDirectory>/*/python to sys.path
    /// NB: can also be used for projects <projectDirectory>/*/python
    static void addPythonModulePathsForPlugins(const std::string& pluginsDirectory);

    /// add module to python context, Init() must have been called before
    static void addModule(const std::string& name, PyMethodDef* methodDef);

    // basic script functions
    static std::string  getError();
    static bool         runString(const std::string& script);
    static bool         runFile( const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0) );

    /// should the future scene loadings reload python modules?
    static void setAutomaticModuleReload( bool );

    /// excluding a module from automatic reload
    static void excludeModuleFromReload( const std::string& moduleName );

    /// to be able to react when a scene is loaded
    struct SceneLoaderListerner : public SceneLoader::Listener
    {
        virtual void rightBeforeLoadingScene(); // possibly unload python modules to force importing their eventual modifications
        static SceneLoaderListerner* getInstance() { static SceneLoaderListerner sceneLoaderListerner; return &sceneLoaderListerner; } // singleton
    private:
        SceneLoaderListerner(){}
    };
};


} // namespace simulation

} // namespace sofa


#endif // PYTHONENVIRONMENT_H
