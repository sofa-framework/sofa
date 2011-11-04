/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "CategoryLibrary.h"

namespace sofa
{

namespace core
{


//-------------------------------------------------------------------------------------------------------
CategoryLibrary::CategoryLibrary( const std::string &categoryName): name(categoryName)
{
}


ComponentLibrary *CategoryLibrary::addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< std::string > &exampleFiles)
{
    //Special case of Mapping and MechanicalMapping
    bool isMechanicalMapping = (name == "MechanicalMapping");
    bool isMapping           = (name == "Mapping");

    ComponentLibrary* component = createComponent(componentName, entry, exampleFiles);

    //Add the corresponding templates
    typedef std::list< std::pair< std::string, Creator*> >::iterator IteratorEntry;
    IteratorEntry itTemplate;

    //It exists Mappings only Mechanical or only Visual. So, we must add the component if only a creator is available for the current category
    bool componentCreationPossible=false;
    //read all the template possible, and remove unused (for Mapping processing)
    std::list<std::string> templates;
    for (itTemplate=entry->creatorList.begin(); itTemplate!= entry->creatorList.end(); itTemplate++)
    {
        const std::string &templateName = itTemplate->first;
        //If the component corresponds to a MechanicalMapping, we must remove the template related to the visual mapping
        if (isMechanicalMapping)
        {
            const std::string nonMechanical = templateName.substr(0,7);
            if (nonMechanical == "Mapping") continue;
        }
        //If the component corresponds to a Mapping, we must remove the template related to the Mechanical Mapping
        else if (isMapping)
        {
            const std::string mechanical    = templateName.substr(0,17);
            if (mechanical == "MechanicalMapping") continue;
        }
        componentCreationPossible=true;
        //component->addTemplate(itTemplate->first);
        if (templateName == (entry->defaultTemplate.empty() ? std::string("Vec3d") : entry->defaultTemplate))
            templates.push_front(templateName); // make sure the default template is first
        else
            templates.push_back(templateName);
    }
    for (std::list<std::string>::const_iterator it = templates.begin(); it != templates.end(); ++it)
        component->addTemplate(*it);
    component->endConstruction();

    //If no constructor is available, we delete the component
    if (!componentCreationPossible)
    {
        delete component;
        component=NULL;
    }
    else
        components.push_back(component);

    return component;
}

void CategoryLibrary::endConstruction()
{
}


const ComponentLibrary *CategoryLibrary::getComponent( const std::string &categoryName) const
{
    for (VecComponentIterator it=components.begin(); it != components.end(); ++it)
    {
        if ((*it)->getName().find(categoryName) != std::string::npos)
            return *it;
    }
    return NULL;
}

}
}
