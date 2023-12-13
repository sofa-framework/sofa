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
getLocalObjects(sofa::simulation::Node& node)
{
    SOFA_UNUSED(node);
    static_assert(false, "Object type is not known");
    return {};
}

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

template<class ObjectType>
class NodeIterator
{
    using container = std::conditional_t<trait::is_single_v<ObjectType>,
                                         NodeSingle<ObjectType>,
                                         NodeSequence<ObjectType, trait::is_strong_v<ObjectType>>>;

    using data_iterator = typename container::iterator;
    using const_data_iterator = typename container::iterator;

    sofa::simulation::Node* m_rootNode { nullptr };
    std::stack<sofa::simulation::Node*> m_stack;

    data_iterator m_currentDataIterator;
    data_iterator m_currentDataEndIterator;
    bool m_isNull { false };

    void findFirstObject()
    {
        auto* current = m_stack.top();
        m_stack.pop();

        for (auto& node : current->child)
        {
            m_stack.push(node.get());
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

    ObjectType* operator*() { return *m_currentDataIterator; }
    const ObjectType* operator*() const { return *m_currentDataIterator; }

    [[nodiscard]] ObjectType* const ptr() const
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
            return m_currentDataIterator->get();
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

template<class ObjectType>
struct SceneGraphObjectTraversal
{
    using iterator = NodeIterator<ObjectType>;

    explicit SceneGraphObjectTraversal(sofa::simulation::Node* root)
        : m_root{root}
    {}

    SceneGraphObjectTraversal() = delete;

    iterator begin() const
    {
        return iterator{m_root};
    }

    iterator end() const
    {
        return iterator{nullptr};
    }

private:
    sofa::simulation::Node* m_root { nullptr };
};

}
