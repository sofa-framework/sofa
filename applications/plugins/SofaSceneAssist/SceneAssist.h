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

#ifndef SOFASCENEASSIST_SCENEASSIST_H
#define SOFASCENEASSIST_SCENEASSIST_H

#include <SofaSimulationGraph/DAGNode.h>
#include <sofa/core/BehaviorModel.h>
#include <SofaSceneAssist/config.h>

namespace sofa
{

namespace _sceneassist_
{

using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::BaseContext ;
using sofa::core::objectmodel::BaseNode ;
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::BaseObjectDescription ;
using sofa::simulation::Node ;

typedef std::map<std::string, std::string> Dict ;

class SceneAssist
{
public:
    static Node::SPtr createNode(Node* node, const std::string name) ;
    static Node::SPtr createNode(BaseNode::SPtr& node, const std::string name) ;
    static Node::SPtr createNode(Node::SPtr& node, const std::string name) ;
    static Node::SPtr createNode(BaseContext::SPtr& context, const std::string name) ;

    static BaseObject::SPtr createObject(Node::SPtr& node, const std::string type, const Dict& kw = Dict()) ;
    static BaseObject::SPtr createObject(BaseContext* context, const std::string type, const Dict& kw = Dict()) ;
    static BaseObject::SPtr createObject(Base* context, const std::string type, const Dict& kw = Dict()) ;

    static void deleteObjectFrom(BaseContext* context, BaseObject* object) ;
    static void deleteNode(Node* parent, Node* node) ;
    static void deleteNode(Node* parent, BaseNode::SPtr& node) ;
    static void deleteNode(Node::SPtr parent, BaseNode* node) ;
} ;

} /// namespace _sceneassist_

using _sceneassist_::Dict ;
using _sceneassist_::SceneAssist ;

} /// namespace sofa

#endif // SOFASCENEASSIST_SCENEASSIST_H
