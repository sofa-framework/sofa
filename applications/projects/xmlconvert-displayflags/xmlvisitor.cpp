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
            if(attribute_value == "1" ) flags->m_showAll.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showAll.setValue(tristate::false_value);
        }
        if( attribute_name == "showVisual" )
        {
            if(attribute_value == "1" )  flags->m_showVisual.setValue(tristate::true_value);
            if(attribute_value == "0" )  flags->m_showVisual.setValue(tristate::false_value);
        }
        if( attribute_name == "showVisualModels")
        {
            if(attribute_value == "1" )  flags->m_showVisualModels.setValue(tristate::true_value);
            if(attribute_value == "0" )  flags->m_showVisualModels.setValue(tristate::false_value);
        }
        if( attribute_name == "showBehavior")
        {
            if(attribute_value == "1" )  flags->m_showBehavior.setValue(tristate::true_value);
            if(attribute_value == "0" )  flags->m_showBehavior.setValue(tristate::false_value);
        }
        if( attribute_name == "showForceFields")
        {
            if(attribute_value == "1" ) flags->m_showForceFields.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showForceFields.setValue(tristate::false_value);
        }
        if( attribute_name == "showInteractionForceFields" )
        {
            if(attribute_value == "1" ) flags->m_showForceFields.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showForceFields.setValue(tristate::false_value);
        }
        if( attribute_name == "showBehaviorModels")
        {
            if(attribute_value == "1" ) flags->m_showBehaviorModels.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showBehaviorModels.setValue(tristate::false_value);
        }
        if( attribute_name == "showCollision" )
        {
            if(attribute_value == "1" ) flags->m_showCollision.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showCollision.setValue(tristate::false_value);
        }
        if( attribute_name == "showCollisionModels" )
        {
            if(attribute_value == "1" ) flags->m_showCollisionModels.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showCollisionModels.setValue(tristate::false_value);
        }
        if( attribute_name == "showBoundingCollisionModels")
        {
            if(attribute_value == "1") flags->m_showBoundingCollisionModels.setValue(tristate::true_value);
            if(attribute_value == "0") flags->m_showBoundingCollisionModels.setValue(tristate::false_value);
        }
        if( attribute_name == "showMapping")
        {
            if(attribute_value == "1" ) flags->m_showMapping.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showMapping.setValue(tristate::false_value);
        }
        if( attribute_name == "showMappings")
        {
            if(attribute_value == "1" ) flags->m_showVisualMappings.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showVisualMappings.setValue(tristate::false_value);
        }
        if( attribute_name == "showMechanicalMappings")
        {
            if(attribute_value == "1" ) flags->m_showMechanicalMappings.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showMechanicalMappings.setValue(tristate::false_value);
        }
        if( attribute_name == "showWireFrame" )
        {
            if(attribute_value == "1" ) flags->m_showWireframe.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showWireframe.setValue(tristate::false_value);
        }
        if( attribute_name == "showNormals" )
        {
            if(attribute_value == "1" ) flags->m_showNormals.setValue(tristate::true_value);
            if(attribute_value == "0" ) flags->m_showNormals.setValue(tristate::false_value);
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
        std::cerr << "Could not find displayFlags for element : " << origin << std::endl;
        return;
    }
    if(it_parent == map_displayFlags.end() )
    {
        std::cerr << "Could not find displayFlags for element : " << parent_element << std::endl;
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
    if( flags.m_showVisualModels.state().state == tristate::false_value )
        flags.m_showVisualModels.setValue(tristate::neutral_value);
    if( flags.m_showBehaviorModels.state().state == tristate::false_value )
        flags.m_showBehaviorModels.setValue(tristate::neutral_value);
    if( flags.m_showCollisionModels.state().state == tristate::false_value )
        flags.m_showCollisionModels.setValue(tristate::neutral_value);
    if( flags.m_showBoundingCollisionModels.state().state == tristate::false_value )
        flags.m_showBoundingCollisionModels.setValue(tristate::neutral_value);
    if( flags.m_showVisualMappings.state().state == tristate::false_value )
        flags.m_showVisualMappings.setValue(tristate::neutral_value);
    if( flags.m_showMechanicalMappings.state().state == tristate::false_value )
        flags.m_showMechanicalMappings.setValue(tristate::neutral_value);
    if( flags.m_showForceFields.state().state == tristate::false_value )
        flags.m_showForceFields.setValue(tristate::neutral_value);
    if( flags.m_showInteractionForceFields.state().state == tristate::false_value )
        flags.m_showInteractionForceFields.setValue(tristate::neutral_value);
    if( flags.m_showWireframe.state().state == tristate::false_value  )
        flags.m_showWireframe.setValue(tristate::neutral_value);
    if( flags.m_showNormals.state().state == tristate::false_value )
        flags.m_showNormals.setValue(tristate::neutral_value);
}

} // namespace xml

} // namespace sofa
