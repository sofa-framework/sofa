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
#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/simulation/Node.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/simulation/XMLPrintVisitor.h>
using sofa::simulation::XMLPrintVisitor ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

namespace sofa::simpleapi
{

bool importPlugin(const std::string& name)
{
    auto& pluginManager = sofa::helper::system::PluginManager::getInstance();

    const auto status = pluginManager.loadPlugin(name);
    if(status == PluginManager::PluginLoadStatus::SUCCESS)
    {
        sofa::core::ObjectFactory::getInstance()->registerObjectsFromPlugin(name);
    }
    return status == PluginManager::PluginLoadStatus::SUCCESS || status == PluginManager::PluginLoadStatus::ALREADY_LOADED;
}

void dumpScene(Node::SPtr root)
{
    XMLPrintVisitor p(sofa::core::execparams::defaultInstance(), std::cout) ;
    p.execute(root.get()) ;
}

Simulation::SPtr createSimulation(const std::string& type)
{
    if(type!="DAG")
    {
        msg_error("SimpleApi") << "Unable to create simulation of type '"<<type<<"'. Supported type is ['DAG']";
        return nullptr ;
    }

    return std::make_shared<simulation::graph::DAGSimulation>();
}


Node::SPtr createRootNode(Simulation::SPtr s, const std::string& name,
                                              const std::map<std::string, std::string>& params)
{
    return createRootNode(s.get(), name, params);
}

NodeSPtr createRootNode(Simulation* s, const std::string& name,
    const std::map<std::string, std::string>& params)
{
    Node::SPtr root = s->createNewNode(name) ;

    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second);
    }
    root->parse(&desc) ;

    return root ;
}

BaseObject::SPtr createObject(Node::SPtr parent, BaseObjectDescription& desc)
{
    /// Create the object.
    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(parent.get(), &desc);
    if (obj == nullptr)
    {
        std::stringstream msg;
        msg << "Component '" << desc.getName() << "' of type '" << desc.getAttribute("type","") << "' failed:" << msgendl ;
        for (const auto& error : desc.getErrors())
        {
            msg << " " << error << msgendl ;
        }
        msg_error(parent.get()) << msg.str() ;
        return nullptr;
    }

    return obj ;
}

BaseObject::SPtr createObject(Node::SPtr parent, const std::string& type, const std::map<std::string, std::string>& params)
{
    /// temporarily, the name is set to the type name.
    /// if a "name" parameter is provided, it will overwrite it.
    BaseObjectDescription desc(type.c_str(),type.c_str());
    for(const auto& [dataName, dataValue] : params)
    {
        desc.setAttribute(dataName.c_str(), dataValue);
    }

    return createObject(parent, desc);
}

Node::SPtr createChild(Node::SPtr node, const std::string& name, const std::map<std::string, std::string>& params)
{
    BaseObjectDescription desc(name.c_str(), "Node");
    for(auto& kv : params)
    {
        desc.setAttribute(kv.first.c_str(), kv.second);
    }
    return createChild(node, desc);
}

Node::SPtr createChild(Node::SPtr node, BaseObjectDescription& desc)
{
    Node::SPtr tmp = node->createChild(desc.getName());
    tmp->parse(&desc);
    return tmp;
}

Node::SPtr createNode(const std::string& name)
{
    return core::objectmodel::New<Node>(name);
}

} // namespace sofa::simpleapi
