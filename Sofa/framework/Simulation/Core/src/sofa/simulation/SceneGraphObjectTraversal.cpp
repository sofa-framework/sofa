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
#define SOFA_SIMULATION_SCENEGRAPHOBJECTTRAVERSAL_CPP
#include <sofa/simulation/SceneGraphObjectTraversal.h>

namespace sofa::simulation
{

template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::BaseObject>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BehaviorModel>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseMapping>;

template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::OdeSolver>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::ConstraintSolver>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseLinearSolver>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::BaseTopologyObject>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseForceField>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseInteractionForceField>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseProjectiveConstraintSet>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseConstraintSet>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::ContextObject>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::objectmodel::ConfigurationSetting>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::Shader>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualModel>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualManager>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::CollisionModel>;

template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseAnimationLoop>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::visual::VisualLoop>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::Topology>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::topology::BaseMeshTopology>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseState>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseMechanicalState>;
// template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::BaseMapping>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::behavior::BaseMass>;
template struct SOFA_SIMULATION_CORE_API SceneGraphObjectTraversal<sofa::core::collision::Pipeline>;

}
