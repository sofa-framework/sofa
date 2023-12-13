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

#include <sofa/simulation/Node.h>

namespace sofa::simulation
{

namespace trait
{
template<class T> struct is_strong : std::false_type {};
template<> struct is_strong<sofa::core::objectmodel::BaseObject> : std::true_type {};

template< class T >
inline constexpr bool is_strong_v = is_strong<T>::value;

template<class T> struct is_single : std::false_type {};
template<> struct is_single<sofa::core::behavior::BaseAnimationLoop> : std::true_type {};
template<> struct is_single<sofa::core::visual::VisualLoop> : std::true_type {};
template<> struct is_single<sofa::core::topology::Topology> : std::true_type {};
template<> struct is_single<sofa::core::topology::BaseMeshTopology> : std::true_type {};
template<> struct is_single<sofa::core::BaseState> : std::true_type {};
template<> struct is_single<sofa::core::behavior::BaseMechanicalState> : std::true_type {};
// template<> struct is_single<sofa::core::BaseMapping> : std::true_type {};
template<> struct is_single<sofa::core::behavior::BaseMass> : std::true_type {};
template<> struct is_single<sofa::core::collision::Pipeline> : std::true_type {};

template< class T >
inline constexpr bool is_single_v = is_single<T>::value;
}

template<class ObjectType>
std::conditional_t<trait::is_single_v<ObjectType>,
    NodeSingle<ObjectType>,
    NodeSequence<ObjectType, trait::is_strong_v<ObjectType>>
>&
getLocalObjects(sofa::simulation::Node& node);

#define GETLOCALOBJECTS(type, name) \
template<> \
inline std::conditional_t<trait::is_single_v<type>,\
    NodeSingle<type>,\
    NodeSequence<type, trait::is_strong_v<type>>\
>& getLocalObjects<type>(sofa::simulation::Node& node) \
{\
    return node.name;\
}

GETLOCALOBJECTS(sofa::core::objectmodel::BaseObject, object)
GETLOCALOBJECTS(sofa::core::BehaviorModel, behaviorModel)
GETLOCALOBJECTS(sofa::core::BaseMapping, mapping)

GETLOCALOBJECTS(sofa::core::behavior::OdeSolver, solver)
GETLOCALOBJECTS(sofa::core::behavior::ConstraintSolver, constraintSolver)
GETLOCALOBJECTS(sofa::core::behavior::BaseLinearSolver, linearSolver)
GETLOCALOBJECTS(sofa::core::topology::BaseTopologyObject, topologyObject)
GETLOCALOBJECTS(sofa::core::behavior::BaseForceField, forceField)
GETLOCALOBJECTS(sofa::core::behavior::BaseInteractionForceField, interactionForceField)
GETLOCALOBJECTS(sofa::core::behavior::BaseProjectiveConstraintSet, projectiveConstraintSet)
GETLOCALOBJECTS(sofa::core::behavior::BaseConstraintSet, constraintSet)
GETLOCALOBJECTS(sofa::core::objectmodel::ContextObject, contextObject)
GETLOCALOBJECTS(sofa::core::objectmodel::ConfigurationSetting, configurationSetting)
GETLOCALOBJECTS(sofa::core::visual::Shader, shaders)
GETLOCALOBJECTS(sofa::core::visual::VisualModel, visualModel)
GETLOCALOBJECTS(sofa::core::visual::VisualManager, visualManager)
GETLOCALOBJECTS(sofa::core::CollisionModel, collisionModel)

GETLOCALOBJECTS(sofa::core::behavior::BaseAnimationLoop, animationManager)
GETLOCALOBJECTS(sofa::core::visual::VisualLoop, visualLoop)
GETLOCALOBJECTS(sofa::core::topology::Topology, topology)
GETLOCALOBJECTS(sofa::core::topology::BaseMeshTopology, meshTopology)
GETLOCALOBJECTS(sofa::core::BaseState, state)
GETLOCALOBJECTS(sofa::core::behavior::BaseMechanicalState, mechanicalState)
// GETLOCALOBJECTS(sofa::core::BaseMapping, mechanicalMapping)
GETLOCALOBJECTS(sofa::core::behavior::BaseMass, mass)
GETLOCALOBJECTS(sofa::core::collision::Pipeline, collisionPipeline)

#undef GETLOCALOBJECTS

/**
 * @brief Iterator for preorder traversing nodes in a scene graph containing objects of a specific type.
 *
 * This iterator is designed to be used with the `SceneGraphObjectTraversal` class to iterate through nodes
 * in a scene graph that contain objects of a specified type (`ObjectType`).
 *
 * @tparam ObjectType The type of objects to traverse in the scene graph.
 */
template<class ObjectType>
class NodeIterator
{
public:
    using container = std::conditional_t<trait::is_single_v<ObjectType>,
                                         NodeSingle<ObjectType>,
                                         NodeSequence<ObjectType, trait::is_strong_v<ObjectType>>>;
    using value_type = ObjectType*;

    using data_iterator = typename container::iterator;
    using const_data_iterator = typename container::iterator;

private:

    sofa::simulation::Node* m_rootNode { nullptr };
    std::stack<sofa::simulation::Node*> m_stack;

    data_iterator m_currentDataIterator;
    data_iterator m_currentDataEndIterator;
    bool m_isNull { false };

    /// Recursive function traversing the graph, searching for an object of
    /// type ObjectType
    void findFirstObject()
    {
        auto* current = m_stack.top();
        m_stack.pop();

        for (auto it = current->child.rbegin(); it != current->child.rend(); ++it)
        {
            m_stack.push(it->get());
        }

        if (auto& objects = getLocalObjects<ObjectType>(*current); !objects.empty())
        {
            m_currentDataIterator = objects.begin();
            m_currentDataEndIterator = objects.end();
            m_isNull = false;
        }
        else
        {
            for (auto& node : current->child)
            {
                SOFA_UNUSED(node);
                findFirstObject();
            }
        }
    }

    void increment()
    {
        if (m_rootNode == nullptr)
            return;

        ++m_currentDataIterator;
        if (m_currentDataIterator == m_currentDataEndIterator)
        {
            if (m_stack.empty())
            {
                m_currentDataIterator = data_iterator{};
                m_currentDataEndIterator = data_iterator{};
                m_isNull = true;
                return;
            }

            findFirstObject();
        }
    }

public:

    explicit NodeIterator(sofa::simulation::Node* current)
        : m_rootNode{current}, m_isNull(true)
    {
        if (current != nullptr)
        {
            m_stack.push(current);
            findFirstObject();
        }
    }

    value_type operator*() { return ptr(); }
    value_type operator*() const { return ptr(); }

    [[nodiscard]] value_type ptr() const
    {
        if (m_rootNode == nullptr)
        {
            return nullptr;
        }
        if (m_currentDataIterator == data_iterator{})
        {
            return nullptr;
        }
        if constexpr (trait::is_single_v<ObjectType>)
        {
            return *m_currentDataIterator;
        }
        else
        {
            if constexpr (trait::is_strong_v<ObjectType>)
            {
                return m_currentDataIterator->get();
            }
            else
            {
                return *m_currentDataIterator;
            }
        }
    }

    /// post-increment
    NodeIterator operator++(int)
    {
        NodeIterator it = *this;
        increment();
        return it;
    }

    /// pre-increment
    NodeIterator& operator++()
    {
        increment();
        return *this;
    }

    bool operator==(const NodeIterator& it)
    {
        if (m_isNull && it.m_isNull)
        {
            return true;
        }
        return m_isNull == it.m_isNull && m_currentDataIterator == it.m_currentDataIterator;
    }
    bool operator!=(const NodeIterator& it)
    {
        return !operator==(it);
    }
};

#if !defined(SOFA_SIMULATION_NODEITERATOR_CPP)
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::BaseObject>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BehaviorModel>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseMapping>;

extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::OdeSolver>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::ConstraintSolver>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseLinearSolver>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::BaseTopologyObject>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseForceField>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseInteractionForceField>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseProjectiveConstraintSet>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseConstraintSet>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::ContextObject>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::ConfigurationSetting>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::Shader>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualModel>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualManager>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::CollisionModel>;

extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseAnimationLoop>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualLoop>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::Topology>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::BaseMeshTopology>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseState>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseMechanicalState>;
// extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseMapping>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseMass>;
extern template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::collision::Pipeline>;
#endif
}
