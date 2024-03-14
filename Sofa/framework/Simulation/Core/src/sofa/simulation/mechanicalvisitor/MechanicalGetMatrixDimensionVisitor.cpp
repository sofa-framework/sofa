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

#include <sofa/simulation/mechanicalvisitor/MechanicalGetMatrixDimensionVisitor.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

namespace sofa::simulation::mechanicalvisitor
{

MechanicalGetMatrixDimensionVisitor::MechanicalGetMatrixDimensionVisitor(const core::ExecParams *params,
                                                                         sofa::Size *const _nbRow,
                                                                         sofa::Size *const _nbCol,
                                                                         sofa::core::behavior::MultiMatrixAccessor *_matrix)
        : BaseMechanicalVisitor(params) , nbRow(_nbRow), nbCol(_nbCol), matrix(_matrix)
{}

Visitor::Result
MechanicalGetMatrixDimensionVisitor::fwdMechanicalState(simulation::Node *, core::behavior::BaseMechanicalState *ms)
{
    //ms->contributeToMatrixDimension(nbRow, nbCol);
    const auto n = ms->getMatrixSize();
    if (nbRow) *nbRow += n;
    if (nbCol) *nbCol += n;
    if (matrix) matrix->addMechanicalState(ms);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalGetMatrixDimensionVisitor::fwdMechanicalMapping(simulation::Node *, core::BaseMapping *mm)
{
    if (matrix) matrix->addMechanicalMapping(mm);
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalGetMatrixDimensionVisitor::fwdMappedMechanicalState(simulation::Node *,
                                                                              core::behavior::BaseMechanicalState *ms)
{
    if (matrix) matrix->addMappedMechanicalState(ms);
    return RESULT_CONTINUE;
}
}