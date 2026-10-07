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

#include <sofa/simulation/config.h>

#include <sofa/core/MultiVecId.h>
#include <sofa/simulation/MappingGraph.h>

namespace sofa::simulation::common
{

/**
 * @brief Differential and geometric operations over a MappingGraph.
 *
 * Provides static operations for pushing forward and pulling back multi-vectors
 * through the hierarchy of mappings defined within a MappingGraph.
 *
 * In physical simulations, primal quantities (coordinates, tangent vectors such as velocities
 * or displacements) are propagated top-to-bottom through pushforward operations (applying
 * the mapping function or its Jacobian $\mathbf{J}$). Dual quantities (cotangent vectors such
 * as forces) are transformed bottom-to-top through pullback operations (using the transpose
 * Jacobian $\mathbf{J}^T$).
 */
class SOFA_SIMULATION_CORE_API DifferentialOperations
{
public:

    /**
     * @brief Push forward primal coordinates top-down through the mapping graph.
     *
     * Traverses the mapping graph from top to bottom and applies each mapping function
     * to transform primal configuration coordinates (e.g. positions) across coordinate frames.
     *
     * @param mappingGraph The mapping graph containing the topology and mappings to traverse.
     * @param mparams Mechanical parameters associated with the current operation.
     * @param coordVectorId Identifier of the coordinate multi-vector to push forward.
     */
    static void pushforwardCoord(
        const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
        core::MultiVecCoordId coordVectorId);

    /**
     * @brief Push forward tangent vectors top-down via the mapping Jacobian.
     *
     * Traverses the mapping graph from top to bottom and applies the Jacobian matrix $\mathbf{J}$
     * of each mapping to propagate tangent quantities (e.g. velocities, displacements $dx$)
     * via Jacobian-vector products ($\mathbf{J}v$).
     *
     * @param mappingGraph The mapping graph containing the topology and mappings to traverse.
     * @param mparams Mechanical parameters associated with the current operation.
     * @param tangentVectorId Identifier of the derivative multi-vector representing tangent quantities to push forward.
     */
    static void pushforwardTangent(
        const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
        core::MultiVecDerivId tangentVectorId);

    /**
     * @brief Pull back cotangent vectors bottom-up via the transpose of the mapping Jacobian.
     *
     * Traverses the mapping graph from bottom to top and applies the transposed Jacobian $\mathbf{J}^T$
     * of each mapping to pull back dual quantities (e.g. forces, momentum) via
     * vector-Jacobian products ($v^T\mathbf{J}$).
     *
     * @param mappingGraph The mapping graph containing the topology and mappings to traverse.
     * @param mparams Mechanical parameters associated with the current operation.
     * @param cotangentVectorId Identifier of the derivative multi-vector representing cotangent quantities to pull back.
     * @param ignoreMappingFlag If true, forces are pulled back through all mappings regardless of
     *                          whether their internal force-mapping flag (`areForcesMapped()`) is enabled.
     */
    static void pullbackCotangent(
        const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
        core::MultiVecDerivId cotangentVectorId,
        bool ignoreMappingFlag = true);

    /**
     * @brief Pull back the differential (tangent variation) of a cotangent vector bottom-up.
     *
     * Computes the total differential of a vector-Jacobian product (VJP), corresponding to
     * the variation of a pulled-back cotangent quantity (such as force variation @f$ df_x = d(\mathbf{J}^T f_y) @f$):
     * @f[
     *     df_x = \mathbf{J}^T df_y + d(\mathbf{J}^T) f_y
     * @f]
     *
     * Traverses the mapping graph from bottom to top and accumulates:
     * 1. The first-order pullback of the cotangent differential via the transpose Jacobian:
     *    @f$ \mathbf{J}^T df_y @f$ (`applyJT()`).
     * 2. When the stiffness factor @c kFactor is non-zero, the geometric stiffness contribution
     *    arising from the non-linear variation of the mapping's Jacobian (a Hessian-vector contraction):
     *    @f$ d(\mathbf{J}^T) f_y @f$ (`applyDJT()`).
     *
     * @param mappingGraph The mapping graph containing the topology and mappings to traverse.
     * @param mparams Mechanical parameters associated with the current operation (including @c kFactor).
     * @param cotangentTangentVectorId Identifier of the derivative multi-vector storing the cotangent
     *                                 tangent variations (e.g. @f$ df @f$) to pull back.
     */
    static void pullbackCotangentTangent(
        const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
        core::MultiVecDerivId cotangentTangentVectorId);
};

}
