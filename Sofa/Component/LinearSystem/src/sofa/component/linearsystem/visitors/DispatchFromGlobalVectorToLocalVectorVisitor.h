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

#include <sofa/simulation/BaseMechanicalVisitor.h>
#include <sofa/component/linearsystem/MappingGraph.h>

namespace sofa::component::linearsystem
{

/**
 * Copy the values stored in a global vector to the local vectors stored in teach BaseMechanicalState
 * The copy location is based on a built MappingGraph.
 */
class SOFA_COMPONENT_LINEARSYSTEM_API DispatchFromGlobalVectorToLocalVectorVisitor : public simulation::BaseMechanicalVisitor
{
public:

    DispatchFromGlobalVectorToLocalVectorVisitor(
        const core::ExecParams* params,
        const MappingGraph& mappingGraph,
        sofa::core::MultiVecId dst,
        linearalgebra::BaseVector * globalVector);

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "DispatchFromGlobalVectorToLocalVectorVisitor"; }

protected:

    /// The vector id to copy from the global vector
    sofa::core::MultiVecId m_dst;

    /// The global vector containing the values to be copied
    sofa::linearalgebra::BaseVector *m_globalVector { nullptr};

    /// Structure used to identify where in the global vector the local vectors will be copied from
    const MappingGraph& m_mappingGraph;
};

} // namespace sofa::component::linearsystem
