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

namespace sofa::simulation::mechanicalvisitor
{

/** Compute the size of a mechanical matrix (mass or stiffness) of the whole scene */
class SOFA_SIMULATION_CORE_API MechanicalGetMatrixDimensionVisitor : public BaseMechanicalVisitor
{
public:
    sofa::Size* const nbRow;
    sofa::Size* const nbCol;
    sofa::core::behavior::MultiMatrixAccessor* matrix;

    MechanicalGetMatrixDimensionVisitor(
            const core::ExecParams* params, sofa::Size* const _nbRow, sofa::Size* const _nbCol,
            sofa::core::behavior::MultiMatrixAccessor* _matrix = nullptr );

    Result fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms) override;

    Result fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* mm) override;

    Result fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override { return "MechanicalGetMatrixDimensionVisitor"; }

};
}