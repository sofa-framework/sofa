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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/bgl/BglMultiMappingElement.h>
#include <sofa/simulation/common/Node.h>
namespace sofa
{
namespace simulation
{
namespace bgl
{

using namespace sofa::simulation::xml;

BglMultiMappingElement::BglMultiMappingElement(const std::string& name, const std::string& type, BaseElement* parent/* =NULL */)
    :BaseMultiMappingElement(name,type,parent)
{

}

void BglMultiMappingElement::updateSceneGraph(sofa::core::BaseMapping* multiMapping,
        const helper::vector<simulation::Node*>& /*ancestorInputs*/,
        helper::vector<simulation::Node*>& otherInputs,
        helper::vector<simulation::Node*>& outputs)
{
    sofa::simulation::Node* currentNode = dynamic_cast<sofa::simulation::Node*>(multiMapping->getContext());

    helper::vector<sofa::simulation::Node*>::iterator iterNode;

    /* add the currentNode to the filteredInputNodes child list */
    for( iterNode = otherInputs.begin(); iterNode != otherInputs.end(); ++iterNode )
    {
        (*iterNode)->addChild(currentNode);
    }

    /* add the outputNodes to the currentNode child list */
    for( iterNode = outputs.begin(); iterNode != outputs.end(); ++iterNode)
    {
        if( *iterNode != currentNode)
        {
            currentNode->addChild(*iterNode);
        }
    }

}

SOFA_DECL_CLASS(BglMultiMappingElement)

helper::Creator<sofa::simulation::xml::BaseElement::NodeFactory, BglMultiMappingElement> BglNodeMultiMappingClass("BglNodeMultiMapping");

const char* BglMultiMappingElement::getClass() const
{
    return BglNodeMultiMappingClass.c_str();
}

}

}

}
