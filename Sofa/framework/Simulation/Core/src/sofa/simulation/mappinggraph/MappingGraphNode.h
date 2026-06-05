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

#include <sofa/simulation/mappinggraph/BaseMappingGraphNode.h>

namespace sofa::simulation
{

/**
 * @brief Template class representing a graph node associated with any SOFA component.
 *
 * This wrapper allows the `MappingGraph` to manage nodes for various types of
 * components (e.g., MechanicalState, ForceField) polymorphically while maintaining
 * type safety and providing standard graph node interfaces.
 *
 * @tparam TComponent The actual SOFA component class pointer type.
 */
template<class TComponent>
class MappingGraphNode : public BaseMappingGraphNode
{
public:
    using SPtr = std::shared_ptr<MappingGraphNode>;
    friend class MappingGraph;

    /**
     * @brief Constructs a node wrapper for the given component.
     * @param s A shared pointer to the component instance.
     */
    explicit MappingGraphNode(typename TComponent::SPtr s)
        : m_component(std::move(s))
    {}

    /**
     * @brief Implements accept by calling visit on the wrapped component.
     * @param visitor The concrete visitor implementation.
     */
    void accept(MappingGraphVisitor& visitor) const override
    {
        if (m_component)
        {
            visitor.visit(*m_component);
        }
    }

    /**
     * @brief Returns the name of the wrapped component.
     * @return The name string from the component.
     */
    std::string getName() const override
    {
        return m_component->getName();
    }

    NodeType getType() const override
    {
        if constexpr (std::is_base_of_v<core::behavior::BaseMechanicalState, TComponent>)
            return NodeType::MechanicalState;
        else if constexpr (std::is_base_of_v<core::BaseMapping, TComponent>)
            return NodeType::Mapping;
        else
            return NodeType::Component;
    }

private:
    typename TComponent::SPtr m_component; ///< The actual SOFA component instance pointer.
};

}
