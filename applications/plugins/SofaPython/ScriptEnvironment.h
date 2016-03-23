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
#ifndef SCRIPTENVIRONMENT_H
#define SCRIPTENVIRONMENT_H

//#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{


class ScriptEnvironment
{
public:
    // nodes initialization stuff
    static void     nodeCreatedByScript(Node* node);        // to be called each time a new node is created by a script.
    static void     initScriptNodes();                      // to be called after each call to a script function.
    static bool     isNodeCreatedByScript(Node* node);

    static void     enableNodeQueuedInit(bool enable);
};


} // namespace core

} // namespace sofa



#endif // SCRIPTENVIRONMENT_H
