/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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


//#include <sofa/simulation/tree/GNode.h>
#include "PythonCommon.h"
#include "Binding.h"
#include <sofa/SofaPython.h>
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

    // helper functions
    //static sofa::simulation::tree::GNode::SPtr  initGraphFromScript( const char *filename );        // returns root node

    // basic script functions
    static PyObject*    importScript( const char *filename, const std::vector<std::string>& arguments=std::vector<std::string>(0) );
    //static bool         initGraph(PyObject *script, sofa::simulation::tree::GNode::SPtr graphRoot);  // calls the method "initGraph(root)" of the script
};


} // namespace core

} // namespace sofa


#endif // PYTHONENVIRONMENT_H
