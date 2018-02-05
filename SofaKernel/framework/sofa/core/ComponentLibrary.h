/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENTLIBRARY_H
#define SOFA_COMPONENTLIBRARY_H

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace core
{

typedef sofa::core::ObjectFactory::ClassEntry ClassEntry;

/**
 *  \brief An Generic Component of the Sofa Library
 *
 *  It contains all the information related to a Sofa component: its name, the templates available, a description of it, its creator, ...
 *  This Interface is used for the Modeler mainly.
 *
 */
class SOFA_CORE_API ComponentLibrary
{
public:
    ComponentLibrary(const std::string& componentName, const std::string& categoryName, ClassEntry::SPtr entry, const std::vector< std::string >& exampleFiles);
    virtual ~ComponentLibrary() {};

    virtual void addTemplate( const std::string& templateName);
    virtual void endConstruction();
    virtual void setDisplayed(bool ) {};

    const std::string& getName()                     const { return name;}
    const std::string& getDescription()              const { return description;}
    const std::string& getCategory()                 const { return categoryName;}
    const std::vector< std::string >& getTemplates() const { return templateName;}
    const ClassEntry::SPtr  getEntry()               const { return entry;}

protected:
    //--------------------------------------------
    //Sofa information
    std::string name;
    std::vector< std::string > templateName;
    std::string description;
    std::string categoryName;
    ClassEntry::SPtr entry;
};
}
}

#endif
