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

#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/MappingGraph.h>

namespace sofa::simulation::common
{

/**
 * @brief Provides mechanical operations functionality using a MappingGraph.
 *
 * This class extends {@link MechanicalOperations}, providing methods that adapt core
 * mechanical computations to use the topological structure provided by a {@link MappingGraph}.
 *
 * The standard {@link MechanicalOperations} typically relies on scene graph visitors, which may
 * fail or yield incorrect results for complex mappings where the required execution order
 * follows the specific dependencies defined in a mapping graph. Therefore, mechanical operations,
 * especially those involving kinematic mappings, must be performed by traversing and respecting
 * the structure of the MappingGraph to ensure computational consistency and correctness.
 *
 * Note: The use of {@link MappingGraphMechanicalOperations} is strongly recommended. In future
 * versions, reliance on the scene graph will be deprecated, and this class will become the
 * primary and exclusive mechanism for performing such operations.
 */
class SOFA_SIMULATION_CORE_API MappingGraphMechanicalOperations : public MechanicalOperations
{
public:
    using MechanicalOperations::MechanicalOperations;

    /// Apply projective constraints to the given vector
    void projectResponse(const MappingGraph& mappingGraph, core::MultiVecDerivId dx, double** W = nullptr);
    using MechanicalOperations::projectResponse;

    /// Compute the current force (given the latest propagated position and velocity)
    void computeForce(const MappingGraph& mappingGraph, core::MultiVecDerivId result, bool clearForceBefore, bool accumulateForcesFromMappedStates, TaskScheduler* taskScheduler);
    using MechanicalOperations::computeForce;

    /// accumulate $ df += (m M + b B + k K) velocity $
    void addMBKv(const MappingGraph& mappingGraph, core::MultiVecDerivId df, core::MatricesFactors::M m, core::MatricesFactors::B b, core::MatricesFactors::K k, bool clear = true, bool accumulate = true);
    using MechanicalOperations::addMBKv;
};

}
