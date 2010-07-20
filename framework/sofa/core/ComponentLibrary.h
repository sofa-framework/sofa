/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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
    ComponentLibrary(const std::string& componentName, const std::string& categoryName, ClassEntry* entry, const std::vector< std::string >& exampleFiles);
    virtual ~ComponentLibrary() {};

    virtual void addTemplate( const std::string& templateName);
    virtual void endConstruction();
    virtual void setDisplayed(bool ) {};

    const std::string& getName()                     const { return name;}
    const std::string& getDescription()              const { return description;}
    const std::string& getCategory()                 const { return categoryName;}
    const std::vector< std::string >& getTemplates() const { return templateName;}
    const ClassEntry*  getEntry()                    const { return entry;}

protected:
    //--------------------------------------------
    //Sofa information
    std::string name;
    std::vector< std::string > templateName;
    std::string description;
    std::string categoryName;
    ClassEntry* entry;
};
}
}

#endif
