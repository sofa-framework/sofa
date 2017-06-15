/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "xmlvisitor.h"

using namespace sofa::core::visual;


namespace sofa
{

namespace xml
{


bool DiscoverNodes::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* )
{
    if( element.ValueStr() != std::string("Node") ) return true;
    const TiXmlElement* child;
    bool is_leafnode = true;
    for( child = element.FirstChildElement() ; child != 0; child = child->NextSiblingElement())
    {
        if( child->ValueStr() == std::string("Node"))
        {
            is_leafnode = false;
            break;
        }
    }
    if( is_leafnode)
    {
        //std::cout << element << std::endl;
        leaves.push_back(const_cast<TiXmlElement*>(&element));
    }
    nodes.push_back(const_cast<TiXmlElement*>(&element));
    return true;
}

bool DiscoverDisplayFlagsVisitor::VisitEnter(const TiXmlElement & element, const TiXmlAttribute * attribute)
{

    // skip elements other than Nodes
    if(element.ValueStr() != std::string("Node") ) return true;

    // if this is the first time we discover this node, turn all its flags to neutral
    if( map_displayFlags.find(&element) == map_displayFlags.end() )
    {
        sofa::core::visual::DisplayFlags* flags = new sofa::core::visual::DisplayFlags();
        flags->setShowVisualModels(tristate::neutral_value);
        map_displayFlags[&element] = flags;
    }

    sofa::core::visual::DisplayFlags* flags = map_displayFlags[&element];

    const TiXmlAttribute* current;
    for( current = attribute; current != NULL; current = current->Next() )
    {
        std::string attribute_name( current->Name() );
        std::string attribute_value( current->Value() );

        if( attribute_name == "showAll")
        {
            if(attribute_value == "1" ) flags->setShowAll(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowAll(tristate::false_value);
        }
        if( attribute_name == "showVisual" )
        {
            if(attribute_value == "1" )  flags->setShowVisual(tristate::true_value);
            if(attribute_value == "0" )  flags->setShowVisual(tristate::false_value);
        }
        if( attribute_name == "showVisualModels")
        {
            if(attribute_value == "1" )  flags->setShowVisualModels(tristate::true_value);
            if(attribute_value == "0" )  flags->setShowVisualModels(tristate::false_value);
        }
        if( attribute_name == "showBehavior")
        {
            if(attribute_value == "1" )  flags->setShowBehavior(tristate::true_value);
            if(attribute_value == "0" )  flags->setShowBehavior(tristate::false_value);
        }
        if( attribute_name == "showForceFields")
        {
            if(attribute_value == "1" ) flags->setShowForceFields(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowForceFields(tristate::false_value);
        }
        if( attribute_name == "showInteractionForceFields" )
        {
            if(attribute_value == "1" ) flags->setShowInteractionForceFields(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowInteractionForceFields(tristate::false_value);
        }
        if( attribute_name == "showBehaviorModels")
        {
            if(attribute_value == "1" ) flags->setShowBehaviorModels(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowBehaviorModels(tristate::false_value);
        }
        if( attribute_name == "showCollision" )
        {
            if(attribute_value == "1" ) flags->setShowCollision(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowCollision(tristate::false_value);
        }
        if( attribute_name == "showCollisionModels" )
        {
            if(attribute_value == "1" ) flags->setShowCollisionModels(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowCollisionModels(tristate::false_value);
        }
        if( attribute_name == "showBoundingCollisionModels")
        {
            if(attribute_value == "1") flags->setShowBoundingCollisionModels(tristate::true_value);
            if(attribute_value == "0") flags->setShowBoundingCollisionModels(tristate::false_value);
        }
        if( attribute_name == "showMapping")
        {
            if(attribute_value == "1" ) flags->setShowMapping(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowMapping(tristate::false_value);
        }
        if( attribute_name == "showMappings")
        {
            if(attribute_value == "1" ) flags->setShowMappings(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowMappings(tristate::false_value);
        }
        if( attribute_name == "showMechanicalMappings")
        {
            if(attribute_value == "1" ) flags->setShowMechanicalMappings(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowMechanicalMappings(tristate::false_value);
        }
        if( attribute_name == "showWireFrame" )
        {
            if(attribute_value == "1" ) flags->setShowWireFrame(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowWireFrame(tristate::false_value);
        }
        if( attribute_name == "showNormals" )
        {
            if(attribute_value == "1" ) flags->setShowNormals(tristate::true_value);
            if(attribute_value == "0" ) flags->setShowNormals(tristate::false_value);
        }

    }

    //map_displayFlags[&element] = flags;

    return true;

}

DiscoverDisplayFlagsVisitor::~DiscoverDisplayFlagsVisitor()
{
    std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*>::iterator it_map;
    for(it_map = map_displayFlags.begin(); it_map != map_displayFlags.end(); ++it_map)
    {
        delete it_map->second;
        it_map->second = NULL;
    }

}

void createVisualStyleVisitor(TiXmlElement* origin, const std::map<const TiXmlElement*,sofa::core::visual::DisplayFlags*>& map_displayFlags)
{
    TiXmlNode* parent = origin->Parent();
    TiXmlElement* parent_element = parent->ToElement();
    if( ! parent_element )
    {
        std::map<const TiXmlElement*,DisplayFlags*>::const_iterator it_current;
        it_current = map_displayFlags.find(origin);
        if(it_current == map_displayFlags.end() )
        {
            std::cerr << "Could not find displayFlags for element : " << origin << std::endl;
            return;
        }
        DisplayFlags& current_flags = *it_current->second;
        if( ! current_flags.isNeutral() )
        {
            if( origin->FirstChildElement("VisualStyle") == NULL )
            {
                //convert_false_to_neutral(current_flags);
                TiXmlElement* visualstyle = new TiXmlElement("VisualStyle");
                std::ostringstream oss;
                oss << current_flags;
                visualstyle->SetAttribute("displayFlags",oss.str());
                TiXmlElement* first_child = origin->FirstChildElement();
                if(first_child) origin->InsertBeforeChild(first_child,*visualstyle);
                else origin->LinkEndChild(visualstyle);
            }
        }
        return;
    }

    std::map<const TiXmlElement*,DisplayFlags*>::const_iterator it_current;
    std::map<const TiXmlElement*,DisplayFlags*>::const_iterator it_parent;
    it_current = map_displayFlags.find(origin);
    it_parent  = map_displayFlags.find(parent_element);

    if(it_current == map_displayFlags.end() )
    {
        std::cerr << "Could not find displayFlags for element : " << *origin << std::endl;
        return;
    }
    if(it_parent == map_displayFlags.end() )
    {
        std::cerr << "Could not find displayFlags for element : " << *parent_element << std::endl;
        return;
    }

    DisplayFlags* parent_flags  = it_parent->second;
    DisplayFlags* current_flags = it_current->second;

    DisplayFlags difference = sofa::core::visual::difference_displayFlags(*parent_flags,*current_flags);
    if( ! difference.isNeutral() )
    {
        if( origin->FirstChildElement("VisualStyle") == NULL )
        {
            TiXmlElement* visualstyle = new TiXmlElement("VisualStyle");
            std::ostringstream oss;
            oss << difference;
            visualstyle->SetAttribute("displayFlags",oss.str());
            TiXmlElement* first_child = origin->FirstChildElement();
            if(first_child) origin->InsertBeforeChild(first_child,*visualstyle);
            else origin->LinkEndChild(visualstyle);

        }
    }
    createVisualStyleVisitor(parent_element,map_displayFlags);
}


void removeShowAttributes(TiXmlElement* node)
{
    node->RemoveAttribute("showAll");
    node->RemoveAttribute("showVisual");
    node->RemoveAttribute("showVisualModels");
    node->RemoveAttribute("showBehavior");
    node->RemoveAttribute("showForceFields");
    node->RemoveAttribute("showInteractionForceFields");
    node->RemoveAttribute("showBehaviorModels");
    node->RemoveAttribute("showCollision");
    node->RemoveAttribute("showCollisionModels");
    node->RemoveAttribute("showBoundingCollisionModels");
    node->RemoveAttribute("showMapping");
    node->RemoveAttribute("showMappings");
    node->RemoveAttribute("showMechanicalMappings");
    node->RemoveAttribute("showWireFrame");
    node->RemoveAttribute("showNormals");
    node->RemoveAttribute("showProcessorColor");

}

void convert_false_to_neutral(DisplayFlags& flags)
{
    if( flags.getShowVisualModels().state == tristate::false_value )
        flags.setShowVisualModels(tristate::neutral_value);
    if( flags.getShowBehaviorModels().state == tristate::false_value )
        flags.setShowBehaviorModels(tristate::neutral_value);
    if( flags.getShowCollisionModels().state == tristate::false_value )
        flags.setShowCollisionModels(tristate::neutral_value);
    if( flags.getShowBoundingCollisionModels().state == tristate::false_value )
        flags.setShowBoundingCollisionModels(tristate::neutral_value);
    if( flags.getShowMappings().state == tristate::false_value )
        flags.setShowMappings(tristate::neutral_value);
    if( flags.getShowMechanicalMappings().state == tristate::false_value )
        flags.setShowMechanicalMappings(tristate::neutral_value);
    if( flags.getShowForceFields().state == tristate::false_value )
        flags.setShowForceFields(tristate::neutral_value);
    if( flags.getShowInteractionForceFields().state == tristate::false_value )
        flags.setShowInteractionForceFields(tristate::neutral_value);
    if( flags.getShowWireFrame().state == tristate::false_value  )
        flags.setShowWireFrame(tristate::neutral_value);
    if( flags.getShowNormals().state == tristate::false_value )
        flags.setShowNormals(tristate::neutral_value);
}

} // namespace xml

} // namespace sofa
