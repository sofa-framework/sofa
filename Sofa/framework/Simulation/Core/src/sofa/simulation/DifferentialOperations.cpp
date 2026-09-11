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
#include <sofa/simulation/DifferentialOperations.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa::simulation::common
{

void DifferentialOperations::pushforwardCoord(
    const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
    core::MultiVecCoordId coordVectorId)
{
    mappingGraph.algorithms.traverseTopDown_([&](core::BaseMapping& mapping)
    {
        mapping.apply(&mparams, coordVectorId, coordVectorId);
    });
}

void DifferentialOperations::pushforwardTangent(
    const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
    core::MultiVecDerivId tangentVectorId)
{
    mappingGraph.algorithms.traverseTopDown_([&](core::BaseMapping& mapping)
    {
        mapping.applyJ(&mparams, tangentVectorId, tangentVectorId);
    });
}

void DifferentialOperations::pullbackCotangent(
    const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
    core::MultiVecDerivId cotangentVectorId, bool ignoreMappingFlag)
{
    mappingGraph.algorithms.traverseBottomUp_([&](core::BaseMapping& mapping)
    {
        if (mapping.areForcesMapped() || ignoreMappingFlag)
        {
            mapping.applyJT(&mparams, cotangentVectorId, cotangentVectorId);
        }
    });
}

void DifferentialOperations::pullbackCotangentTangent(
    const MappingGraph& mappingGraph, const core::MechanicalParams& mparams,
    core::MultiVecDerivId cotangentTangentVectorId)
{
    mappingGraph.algorithms.traverseBottomUp_(
        [&](core::BaseMapping& mapping)
        {
            mapping.applyJT(&mparams, cotangentTangentVectorId, cotangentTangentVectorId);
            if (mparams.kFactor() != 0)
            {
                // ideally, the child cotangent vector must be provided here, instead of implicitly getting it in the mapping
                mapping.applyDJT(&mparams, cotangentTangentVectorId, cotangentTangentVectorId);
            }
        });
}

}  // namespace sofa::simulation::common
