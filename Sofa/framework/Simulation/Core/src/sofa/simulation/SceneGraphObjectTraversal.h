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

#include <sofa/simulation/NodeIterator.h>

namespace sofa::simulation
{

/**
 * @brief Represents a traversal of a scene graph for a specific object type.
 *
 * This class provides a convenient way to traverse a scene graph for objects of a specific type.
 * It uses an iterator, `NodeIterator`, to iterate through the nodes containing objects of the specified type.
 *
 * @tparam ObjectType The type of objects to traverse in the scene graph.
 */
template<class ObjectType>
struct SceneGraphObjectTraversal
{
    using iterator = NodeIterator<ObjectType>;
    using value_type = ObjectType*;

    explicit SceneGraphObjectTraversal(sofa::simulation::Node* root)
        : m_root{root}
    {}

    SceneGraphObjectTraversal() = delete;

    [[nodiscard]] iterator begin() const
    {
        return iterator{m_root};
    }

    [[nodiscard]] iterator end() const
    {
        return iterator{nullptr};
    }

    [[nodiscard]] typename iterator::difference_type size() const
    {
        return std::distance(begin(), end());
    }

private:
    sofa::simulation::Node* m_root { nullptr };
};

#if !defined(SOFA_SIMULATION_SCENEGRAPHOBJECTTRAVERSAL_CPP)
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::BaseObject>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BehaviorModel>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseMapping>;

extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::OdeSolver>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::ConstraintSolver>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseLinearSolver>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::BaseTopologyObject>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseForceField>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseInteractionForceField>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseProjectiveConstraintSet>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseConstraintSet>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::ContextObject>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::ConfigurationSetting>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::Shader>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualModel>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualManager>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::CollisionModel>;

extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseAnimationLoop>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualLoop>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::Topology>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::BaseMeshTopology>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseState>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseMechanicalState>;
// extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseMapping>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseMass>;
extern template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::collision::Pipeline>;
#endif

}
