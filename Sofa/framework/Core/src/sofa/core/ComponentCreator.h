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

#include <sofa/core/config.h>
#include <sofa/core/objectmodel/SPtr.h>
#include <sofa/core/sptr.h>
#include <memory>

namespace sofa::core::objectmodel { class BaseComponent; }

namespace sofa::core
{

/**
 * @brief Base class for component instantiation.
 *
 * This abstract interface is used by ComponentFactory to interact with component creators
 * without knowing the specific concrete type of the component being created.
 */
struct SOFA_CORE_API BaseComponentCreator
{
    virtual ~BaseComponentCreator() = default;

    /**
     * @brief Instantiates a new component.
     * Used by ComponentFactory to generate object instances before attribute parsing.
     * @return A shared pointer to the newly created BaseComponent.
     */
    virtual sofa::core::sptr<sofa::core::objectmodel::BaseComponent> create() const = 0;

    /**
     * @brief Creates a copy of the creator.
     * Ensures the factory registry owns unique instances of creators during registration.
     */
    virtual std::unique_ptr<BaseComponentCreator> clone() const = 0;
};

/**
 * @brief Templated implementation of BaseComponentCreator.
 *
 * Binds a specific component class to the factory mechanism, ensuring components
 * are created using SOFA's memory management (sofa::core::objectmodel::New).
 *
 * @tparam RealComponent The concrete component class to instantiate.
 */
template<class RealComponent>
struct ComponentCreator : public BaseComponentCreator
{
    sofa::core::sptr<sofa::core::objectmodel::BaseComponent> create() const override
    {
        // WARNING:
        // It obliges the class to have a default constructor
        return objectmodel::New<RealComponent>();
    }

    std::unique_ptr<BaseComponentCreator> clone() const override
    {
        return std::make_unique<ComponentCreator>();
    }
};

}
