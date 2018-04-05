/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/************************************************************************************
 * Contributors:                                                                    *
 *    - damien.marchal@univ-lille1.fr                                               *
 ***********************************************************************************/
#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include "SimpleApi.h"
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/simulation/XMLPrintVisitor.h>
using sofa::simulation::XMLPrintVisitor ;

namespace sofa
{
namespace simpleapi
{

void dumpScene(Node::SPtr root)
{
    XMLPrintVisitor p(sofa::core::ExecParams::defaultInstance(), std::cout) ;
    p.execute(root.get()) ;
}

Simulation::SPtr createSimulation(const std::string& type)
{
    if(type!="DAG")
    {
        msg_error("SimpleApi") << "Unable to create simulation of type '"<<type<<"'. Supported type is ['DAG']";
        return nullptr ;
    }

    return new simulation::graph::DAGSimulation() ;
}


Node::SPtr createRootNode(Simulation::SPtr s, const std::string& name,
                                              const std::map<std::string, std::string>& params)
{
    Node::SPtr root = s->createNewNode(name) ;

    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second.c_str());
    }
    root->parse(&desc) ;

    return root ;
}


BaseObject::SPtr createObject(Node::SPtr parent, const std::string& type, const std::map<std::string, std::string>& params)
{
    /// temporarily, the name is set to the type name.
    /// if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type.c_str(),type.c_str());
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second.c_str());
    }

    /// Create the object.
    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(parent.get(), &desc);
    if (obj==0)
    {
        std::stringstream msg;
        msg << "Component '" << desc.getName() << "' of type '" << desc.getAttribute("type","") << "' failed:" << msgendl ;
        for (std::vector< std::string >::const_iterator it = desc.getErrors().begin(); it != desc.getErrors().end(); ++it)
            msg << " " << *it << msgendl ;
        msg_error(parent.get()) << msg.str() ;
        return NULL;
    }
    return obj ;
}

Node::SPtr createChild(Node::SPtr& node, const std::string& name, const std::map<std::string, std::string>& params)
{
    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second.c_str());
    }
    Node::SPtr tmp = node->createChild(name);
    tmp->parse(&desc);
    return tmp;
}

} /// simpleapi
} /// sofa
