/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/core/SofaLibrary.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::core
{

//Automatically create and destroy all the components available: easy way to verify the default constructor and destructor
void SofaLibrary::build( const std::vector< std::string >& examples)
{
    exampleFiles=examples;
    //-----------------------------------------------------------------------
    //Read the content of the Object Factory
    //-----------------------------------------------------------------------
    std::vector<ClassEntry::SPtr> entries;
    sofa::core::ObjectFactory::getInstance()->getAllEntries(entries);
    //Set of categories found in the Object Factory
    std::set< std::string > mainCategories;
    //Data containing all the entries for a given category
    std::multimap< std::string, ClassEntry::SPtr> inventory;

    for (std::size_t i=0; i<entries.size(); ++i)
    {
        //Insert Template specification
        ObjectFactory::CreatorMap::iterator creatorEntry = entries[i]->creatorMap.begin();
        if (creatorEntry != entries[i]->creatorMap.end())
        {
            const objectmodel::BaseClass* baseClass = creatorEntry->second->getClass();
            std::vector<std::string> categories;
            CategoryLibrary::getCategories(baseClass, categories);
            for (std::vector<std::string>::iterator it = categories.begin(); it != categories.end(); ++it)
            {
                mainCategories.insert((*it));
                inventory.insert(std::make_pair((*it), entries[i]));
            }
        }
    }

    //-----------------------------------------------------------------------
    //Using the inventory, Add each component to the Sofa Library
    //-----------------------------------------------------------------------
    std::set< std::string >::iterator itCategory;
    typedef std::multimap< std::string, ClassEntry::SPtr >::iterator IteratorInventory;


    //We add the components category by category
    for (itCategory = mainCategories.begin(); itCategory != mainCategories.end(); ++itCategory)
    {
        const std::string& categoryName = *itCategory;
        IteratorInventory itComponent;

        std::pair< IteratorInventory,IteratorInventory > rangeCategory;
        rangeCategory = inventory.equal_range(categoryName);



        const unsigned int numComponentInCategory = (unsigned int)inventory.count(categoryName);
        CategoryLibrary *category = createCategory(categoryName,numComponentInCategory);

        //Process all the component of the current category, and add them to the group
        for (itComponent=rangeCategory.first; itComponent != rangeCategory.second; ++itComponent)
        {
            const ClassEntry::SPtr entry = itComponent->second;
            const std::string &componentName=entry->className;

            //Add the component to the category
            category->addComponent(componentName, entry, exampleFiles);
        }
        category->endConstruction();
        addCategory(category);
    }
    computeNumComponents();
}

void SofaLibrary::computeNumComponents()
{
    numComponents=0;
    for (std::size_t cat=0; cat<categories.size(); ++cat)
    {
        numComponents += (unsigned int) categories[cat]->getNumComponents();
    }

}

void SofaLibrary::addCategory(CategoryLibrary *category)
{
    categories.push_back(category);
}


std::string SofaLibrary::getComponentDescription( const std::string &componentName ) const
{
    const ComponentLibrary *component = getComponent(componentName);
    if (component) return component->getDescription();
    else return "";
}

const CategoryLibrary *SofaLibrary::getCategory( const std::string &categoryName) const
{
    for (VecCategoryIterator it=categories.begin(); it != categories.end(); ++it)
    {
        if ((*it)->getName().find(categoryName) != std::string::npos)
            return *it;
    }
    return nullptr;
}

const ComponentLibrary *SofaLibrary::getComponent( const std::string &componentName ) const
{
    //Look into all the categories
    for (std::size_t cat=0; cat<categories.size(); ++cat)
    {
        //For each category, look at all the components if one has the name wanted
        const std::vector< ComponentLibrary* > &components = categories[cat]->getComponents();
        for (std::size_t comp=0; comp<components.size(); ++comp)
        {
            if (componentName == components[comp]->getName()) return components[comp];
        }
    }
    return nullptr;
}

void SofaLibrary::clear()
{
    for (std::size_t i=0; i<categories.size(); ++i)
    {
        delete categories[i];
    }
    categories.clear();
}
}
