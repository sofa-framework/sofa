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

#include "SofaLibrary.h"
#include <sofa/core/ObjectFactory.h>


namespace sofa
{
namespace core
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

    for (auto & entrie : entries)
    {
        //Insert Template specification
        ObjectFactory::CreatorMap::iterator creatorEntry = entrie->creatorMap.begin();
        if (creatorEntry != entrie->creatorMap.end())
        {
            const objectmodel::BaseClass* baseClass = creatorEntry->second->getClass();
            std::vector<std::string> categories;
            CategoryLibrary::getCategories(baseClass, categories);
            for (auto & categorie : categories)
            {
                mainCategories.insert(categorie);
                inventory.insert(std::make_pair(categorie, entrie));
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
            ClassEntry::SPtr entry = itComponent->second;
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
    for (auto & categorie : categories)
    {
        numComponents += (unsigned int) categorie->getNumComponents();
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
    for (auto categorie : categories)
    {
        if (categorie->getName().find(categoryName) != std::string::npos)
            return categorie;
    }
    return NULL;
}

const ComponentLibrary *SofaLibrary::getComponent( const std::string &componentName ) const
{
    //Look into all the categories
    for (auto categorie : categories)
    {
        //For each category, look at all the components if one has the name wanted
        const std::vector< ComponentLibrary* > &components = categorie->getComponents();
        for (auto component : components)
        {
            if (componentName == component->getName()) return component;
        }
    }
    return NULL;
}

void SofaLibrary::clear()
{
    for (auto & categorie : categories)
    {
        delete categorie;
    }
    categories.clear();
}

}
}
