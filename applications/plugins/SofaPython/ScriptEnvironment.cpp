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
#include "ScriptEnvironment.h"

namespace sofa
{

namespace simulation
{

static std::list<Node*> m_NodesCreatedByScript;
static bool m_NodeQueuedInitEnabled = true;


// nodes initialization stuff
void ScriptEnvironment::nodeCreatedByScript(Node* node)        // to be called each time a new node is created by a script.
{
    if (!m_NodeQueuedInitEnabled) return;

    m_NodesCreatedByScript.push_back(node);
}

bool ScriptEnvironment::isNodeCreatedByScript(Node* node)        // to be called each time a new node is created by a script.
{
    if (!m_NodeQueuedInitEnabled) return true;

    std::list<Node*>::iterator it = m_NodesCreatedByScript.begin();
    while (it != m_NodesCreatedByScript.end())
    {
        if ((*it)==node)
            return true;
        it++;
    }
    return false;
}

void ScriptEnvironment::initScriptNodes()                      // to be called after each call to a script function.
{
    if (!m_NodeQueuedInitEnabled) return;

    while (!m_NodesCreatedByScript.empty())
    {
        Node *node = m_NodesCreatedByScript.front();
        m_NodesCreatedByScript.pop_front();
        if (!node->isInitialized())
            node->init(sofa::core::ExecParams::defaultInstance());
    }
}

void ScriptEnvironment::enableNodeQueuedInit(bool enable)
{
    m_NodeQueuedInitEnabled = enable;
}




} // namespace simulation

} // namespace sofa

