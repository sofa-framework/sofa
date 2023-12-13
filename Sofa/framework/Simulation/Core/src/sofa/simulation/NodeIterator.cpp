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
#define SOFA_SIMULATION_NODEITERATOR_CPP

#include <sofa/simulation/NodeIterator.h>
#include <sofa/core/collision/Pipeline.h>

namespace sofa::simulation
{
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::BaseObject>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BehaviorModel>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseMapping>;

template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::OdeSolver>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::ConstraintSolver>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseLinearSolver>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::BaseTopologyObject>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseForceField>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseInteractionForceField>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseProjectiveConstraintSet>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseConstraintSet>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::ContextObject>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::objectmodel::ConfigurationSetting>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::Shader>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualModel>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualManager>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::CollisionModel>;

template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseAnimationLoop>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::visual::VisualLoop>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::Topology>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::topology::BaseMeshTopology>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseState>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseMechanicalState>;
// template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::BaseMapping>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::behavior::BaseMass>;
template class SOFA_SIMULATION_CORE_API NodeIterator<sofa::core::collision::Pipeline>;
}
