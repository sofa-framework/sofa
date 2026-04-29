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
#include <memory>
#include <optional>
#include <sofa/simulation/config.h>
#include <sofa/simulation/mappinggraph/MappingGraphVisitor.h>

namespace sofa::simulation
{

template<class Callable, class Component>
struct BaseCallableVisitor : public MappingGraphVisitor
{
    explicit BaseCallableVisitor(const Callable& callable)
    : m_callable(callable)
    {}

    void visit(Component& component) override
    {
        this->m_callable(component);
    }

protected:
    const Callable& m_callable;
};

template<class Callable>
struct GetComponentFromCallable;

template<class Callable> requires std::is_invocable_v<Callable, core::behavior::BaseForceField&>
struct GetComponentFromCallable<Callable>
{
    using type = core::behavior::BaseForceField;
};

template<class Callable> requires std::is_invocable_v<Callable, core::behavior::BaseMass&>
struct GetComponentFromCallable<Callable>
{
    using type = core::behavior::BaseMass;
};

template<class Callable> requires std::is_invocable_v<Callable, core::behavior::BaseMechanicalState&>
struct GetComponentFromCallable<Callable>
{
    using type = core::behavior::BaseMechanicalState;
};

template<class Callable> requires std::is_invocable_v<Callable, core::BaseMapping&>
struct GetComponentFromCallable<Callable>
{
    using type = core::BaseMapping;
};

template<class Callable>
using CallableVisitor = BaseCallableVisitor<Callable, typename GetComponentFromCallable<Callable>::type>;

}
