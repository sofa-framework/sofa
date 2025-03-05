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
 * Copy a vector stored in a BaseMechanicalState into a global vector of type BaseVector.
 * The copy location is based on a built MappingGraph.
 */
class SOFA_COMPONENT_LINEARSYSTEM_API AssembleGlobalVectorFromLocalVectorVisitor : public simulation::BaseMechanicalVisitor
{
public:

    AssembleGlobalVectorFromLocalVectorVisitor(
        const core::ExecParams* params,
        const MappingGraph& mappingGraph,
        sofa::core::ConstMultiVecId src,
        linearalgebra::BaseVector * globalVector);

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "AssembleGlobalVectorVisitorFromLocalVector"; }

protected:

    /// The vector id to copy in the global vector
    sofa::core::ConstMultiVecId m_src;

    /// The global vector where all the local vectors will be copied into
    sofa::linearalgebra::BaseVector *m_globalVector { nullptr};

    /// Structure used to identify where in the global vector the local vectors will be copied into
    const MappingGraph& m_mappingGraph;
};

} // namespace sofa::component::linearsystem
