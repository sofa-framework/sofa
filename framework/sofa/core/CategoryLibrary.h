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
#ifndef SOFA_CATEGORYLIBRARY_H
#define SOFA_CATEGORYLIBRARY_H

#include "ComponentLibrary.h"

namespace sofa
{

namespace core
{


typedef sofa::core::ObjectFactory::Creator    Creator;

/**
 *  \brief An Generic Category of the Sofa Library
 *
 *  It contains all the components available for Sofa corresponding to a given category (force field, mass, mapping...)
 *  This Interface is used for the Modeler mainly.
 *
 */
class SOFA_CORE_API CategoryLibrary
{
public:
    typedef std::vector< ComponentLibrary* > VecComponent;
    typedef VecComponent::const_iterator VecComponentIterator;

public:
    CategoryLibrary( const std::string &categoryName);
    virtual ~CategoryLibrary() {};

    virtual ComponentLibrary *addComponent(const std::string &componentName, ClassEntry* entry, const std::vector< std::string > &exampleFiles);
    virtual void endConstruction();

    const std::string  &getName()          const { return name;}
    const VecComponent &getComponents()    const {return components;}

    const ComponentLibrary *getComponent( const std::string &componentName) const;

    unsigned int getNumComponents() const {return components.size();}

protected:
    virtual ComponentLibrary *createComponent(const std::string &componentName, ClassEntry* entry, const std::vector< std::string > &exampleFiles) {return new ComponentLibrary(componentName, name, entry, exampleFiles);};

    std::string name;
    VecComponent components;
};

}
}

#endif
