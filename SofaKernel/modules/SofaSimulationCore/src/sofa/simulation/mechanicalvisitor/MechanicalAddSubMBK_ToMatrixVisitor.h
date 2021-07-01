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

#include <sofa/simulation/MechanicalVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

/** Accumulate the entries of a mechanical matrix (mass or stiffness) of the whole scene ONLY ON THE subMatrixIndex */
class SOFA_SIMULATION_CORE_API MechanicalAddSubMBK_ToMatrixVisitor : public MechanicalVisitor
{
public:
    const sofa::core::behavior::MultiMatrixAccessor* matrix;
    const type::vector<unsigned> & subMatrixIndex; // index of the point where the matrix must be computed

    MechanicalAddSubMBK_ToMatrixVisitor(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* _matrix, const type::vector<unsigned> & Id);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalAddSubMBK_ToMatrixVisitor"; }

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*ms*/) override;

    Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff) override;

    //Masses are now added in the addMBKToMatrix call for all ForceFields
};
}