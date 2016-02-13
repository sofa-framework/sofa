/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef PYTHONENVIRONMENT_H
#define PYTHONENVIRONMENT_H

#include "PythonCommon.h"
#include "PythonMacros.h"

//#include <sofa/simulation/tree/GNode.h>
#include "Binding.h"
#include <SofaPython/config.h>
#include <vector>
#include <string>

namespace sofa
{

namespace simulation
{

class SOFA_SOFAPYTHON_API PythonEnvironment
{
public:
    static void     Init();
    static void     Release();

    /// Add a path to sys.path, the list of search path for Python modules.
    static void addPythonModulePath(const std::string& path);

    /// Add each line of a file to sys.path
    static void addPythonModulePathsFromConfigFile(const std::string& path);

    /// Add all the directories matching <pluginsDirectory>/*/python to sys.path
    /// NB: can also be used for projects <projectDirectory>/*/python
    static void addPythonModulePathsForPlugins(const std::string& pluginsDirectory);

    // helper functions
    //static sofa::simulation::tree::GNode::SPtr  initGraphFromScript( const char *filename );        // returns root node

    /// add module to python context, Init() must have been called before
    static void addModule(const std::string& name, PyMethodDef* methodDef);

    // basic script functions
    static std::string  getError();
    static bool         runString(const std::string& script);
    static bool         runFile( const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0) );

    //static bool         initGraph(PyObject *script, sofa::simulation::tree::GNode::SPtr graphRoot);  // calls the method "initGraph(root)" of the script

};


} // namespace core

} // namespace sofa


#endif // PYTHONENVIRONMENT_H
