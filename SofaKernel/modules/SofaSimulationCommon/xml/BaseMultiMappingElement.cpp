/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaSimulationCommon/xml/BaseMultiMappingElement.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <SofaSimulationCommon/xml/NodeElement.h>
#include <SofaSimulationCommon/xml/Element.h>

namespace sofa
{
namespace simulation
{
namespace xml
{
BaseMultiMappingElement::BaseMultiMappingElement(const std::string& name, const std::string& type, BaseElement* parent/* =NULL */)
    :ObjectElement(name,type,parent)
{

}

bool BaseMultiMappingElement::initNode()
{
    using namespace core::objectmodel;
    using namespace core;
    bool result = ObjectElement::initNode();


    if( result )
    {

        BaseMapping* multimapping = this->getTypedObject()->toBaseMapping();
        NodeElement* currentNodeElement = dynamic_cast<NodeElement *>(getParent());
        simulation::Node* currentNode =  dynamic_cast<simulation::Node* >( currentNodeElement->getTypedObject() );
        helper::vector<core::BaseState*> inputStates  = multimapping->getFrom();
        helper::vector<core::BaseState*> outputStates = multimapping->getTo();


        helper::vector<core::BaseState*>::iterator iterState;
        helper::vector<simulation::Node*> inputNodes, outputNodes;

        /* get the Nodes corresponding to each input BaseState context */
        for( iterState = inputStates.begin();  iterState != inputStates.end(); ++iterState)
        {
            simulation::Node* node = dynamic_cast< simulation::Node* >((*iterState)->getContext());
            inputNodes.push_back(node);
        }
        /* */
        /* get the Nodes corresponding to each output BaseState context */
        for( iterState = outputStates.begin(); iterState != outputStates.end(); ++iterState)
        {
            simulation::Node* node = dynamic_cast< simulation::Node* >((*iterState)->getContext());
            outputNodes.push_back(node);
        }

        helper::vector<simulation::Node*>::iterator iterNode;
        BaseNode* currentBaseNode;

        /* filter out inputNodes which already belong to the currentNode ancestors */
        helper::vector<simulation::Node*> otherInputNodes;
        helper::vector<simulation::Node*> ancestorInputNodes;
        iterNode = inputNodes.begin();
        currentBaseNode = currentNode;
        for( iterNode = inputNodes.begin(); iterNode != inputNodes.end(); ++iterNode)
        {
            if( !currentBaseNode->hasAncestor(*iterNode) )
            {
                otherInputNodes.push_back(*iterNode);
            }
            else
            {
                ancestorInputNodes.push_back(*iterNode);
            }
        }

        updateSceneGraph(multimapping, ancestorInputNodes, otherInputNodes, outputNodes );

    }

    return result;
}

}
}
}
