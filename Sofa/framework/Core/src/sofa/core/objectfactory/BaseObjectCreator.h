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
#pragma once

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core::objectfactory
{

/// Abstract interface of objects used to create instances of a given type
class BaseObjectCreator
{
public:
    typedef std::shared_ptr<BaseObjectCreator> SPtr;

    virtual ~BaseObjectCreator() { }

    /// Pre-construction check.
    ///
    /// \return true if the object can be created successfully.
    virtual bool canCreate(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg) = 0;

    /// Construction method called by the factory.
    ///
    /// \pre canCreate(context, arg) == true.
    virtual objectmodel::BaseObject::SPtr createInstance(objectmodel::BaseContext* context,
                                                         objectmodel::BaseObjectDescription* arg) = 0;

    /// type_info structure associated with the type of intanciated objects.
    virtual const std::type_info& type() = 0;

    /// BaseClass structure associated with the type of intanciated objects.
    virtual const objectmodel::BaseClass* getClass() = 0;

    /// The name of the library or executable containing the binary code for this component
    virtual const char* getTarget() = 0;

    virtual const char* getHeaderFileLocation() = 0;
};

typedef std::shared_ptr<BaseObjectCreator> BaseObjectCreatorSPtr;
typedef std::map<std::string, BaseObjectCreatorSPtr> BaseObjectCreatorMap;

} // namespace sofa::core
