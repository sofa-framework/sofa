/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <sofa/simulation/DeleteVisitor.h>
using sofa::simulation::DeleteVisitor ;

#include <SofaSceneAssist/SceneAssist.h>
namespace sofa
{
namespace _sceneassist_
{


Node::SPtr SceneAssist::createNode(Node::SPtr& node, const std::string name)
{
    return SceneAssist::createNode(node.get(), name) ;
}

Node::SPtr SceneAssist::createNode(BaseNode::SPtr& node, const std::string name)
{
    return SceneAssist::createNode(dynamic_cast<Node*>(node.get()), name) ;
}

Node::SPtr SceneAssist::createNode(BaseContext::SPtr& context, const std::string name)
{
    return SceneAssist::createNode(dynamic_cast<Node*>(context.get()), name);
}

Node::SPtr SceneAssist::createNode(Node* node, const std::string name)
{
    if(!node)
        return nullptr;

    return node->createChild(name) ;
}

void SceneAssist::deleteNode(Node* parent, Node* node)
{
    if(!node)
        return ;

    if(!parent)
        return ;

    parent->removeChild(node);
}


void SceneAssist::deleteNode(Node* parent, BaseNode::SPtr &node_)
{
    Node* node = dynamic_cast<Node*>(node_.get());
    if(!node)
        return ;

    if(!parent)
        return ;

    parent->removeChild(node);
}



void SceneAssist::deleteNode(Node::SPtr parent, BaseNode* node)
{
    if(!node)
        return ;

    if(!parent)
        return ;

    parent->removeChild(node);
}



BaseObject::SPtr SceneAssist::createObject(Base* context, const std::string type, const Dict& kw)
{
    BaseContext* b = context->toBaseContext() ;
    assert(b) ;

    return SceneAssist::createObject(b, type, kw) ;
}


BaseObject::SPtr SceneAssist::createObject(Node::SPtr& node, const std::string type, const Dict& kw)
{
    return SceneAssist::createObject(node->toBaseContext(), type, kw) ;
}


BaseObject::SPtr SceneAssist::createObject(BaseContext* context, const std::string type, const Dict& kw)
{
    assert(context) ;

    BaseObjectDescription desc(type.c_str(), type.c_str());

    if (kw.size()>0)
    {
        for ( auto& it : kw )
        {
            const std::string& key = it.first ;
            const std::string& value = it.second ;
            desc.setAttribute(key.c_str(), value.c_str());
        }
    }

    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context, &desc);
    if (obj==nullptr)
    {
        std::stringstream tmp ;
        for ( auto& msg : desc.getErrors() )
            tmp << "-" << msg << msgendl;

        msg_error(context) << "Unable to create object of type " << type << msgendl
                           << tmp.str() ;
    }

    for( auto it : desc.getAttributeMap() )
    {
        if (!it.second.isAccessed())
        {
            msg_error(obj.get()) <<"Unused Attribute: '"<<it.first <<"' with value: '" <<(std::string)it.second<<"'" ;
        }
    }

    return obj;
}


void SceneAssist::deleteObjectFrom(BaseContext *context, BaseObject *object)
{
    assert(object);
    assert(context);

    BaseNode* node = context->toBaseNode() ;
    assert(node);

    node->removeObject(object) ;


}

} /// namespace _sceneassist_

} /// namespace sofa
