/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/simulation/graph/config.h>
#include <string>
#include <sstream>
#include <map>

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/fwd.h>

namespace sofa::simpleapi
{

using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::BaseObjectDescription;

using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::NodeSPtr ;

bool SOFA_SIMULATION_GRAPH_API importPlugin(const std::string& name) ;

Simulation::SPtr SOFA_SIMULATION_GRAPH_API createSimulation(const std::string& type="DAG") ;

NodeSPtr SOFA_SIMULATION_GRAPH_API createRootNode( Simulation::SPtr, const std::string& name,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

NodeSPtr SOFA_SIMULATION_GRAPH_API createRootNode( Simulation* s, const std::string& name,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

///@brief Create a sofa object in the provided node.
///The parameter "params" is for passing specific data argument to the created object including the
///object's type.
sofa::core::sptr<BaseObject> SOFA_SIMULATION_GRAPH_API createObject(NodeSPtr node, BaseObjectDescription& params);

///@brief create a sofa object in the provided node of the given type.
///The parameter "params" is for passing specific data argument to the created object.
sofa::core::sptr<BaseObject> SOFA_SIMULATION_GRAPH_API createObject( NodeSPtr node, const std::string& type,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

///@brief create a child to the provided nodeof given name.
///The parameter "params" is for passing specific data argument to the created object.
NodeSPtr SOFA_SIMULATION_GRAPH_API createChild( NodeSPtr node, const std::string& name,
    const std::map<std::string, std::string>& params = std::map<std::string, std::string>{} );

///@brief create a child to the provided node.
///The parameter "params" is for passing specific data argument to the created object (including the node name).
NodeSPtr SOFA_SIMULATION_GRAPH_API createChild(NodeSPtr node, BaseObjectDescription& desc);

///@brief create a child to the provided node.
///The parameter "params" is for passing specific data argument to the created object (including the node name).
NodeSPtr SOFA_SIMULATION_GRAPH_API createNode(const std::string& name);

void SOFA_SIMULATION_GRAPH_API dumpScene(NodeSPtr root) ;

template<class T>
std::string str(const T& t)
{
    std::stringstream s;
    s << t;
    return s.str() ;
}
} // namespace sofa::simpleapi
