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
#include <sofa/simulation/mechanicalvisitor/MechanicalGetConstraintJacobianVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorToBaseVectorVisitor.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
namespace sofa::simulation::mechanicalvisitor
{

MechanicalGetConstraintJacobianVisitor::MechanicalGetConstraintJacobianVisitor(
        const core::ConstraintParams* cparams, linearalgebra::BaseMatrix * _J, const sofa::core::behavior::MultiMatrixAccessor* _matrix)
    : BaseMechanicalVisitor(cparams) , cparams(cparams), J(_J), matrix(_matrix), offset(0)
{}

MechanicalGetConstraintJacobianVisitor::Result MechanicalGetConstraintJacobianVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
{
    if (matrix) offset = matrix->getGlobalOffset(ms);

    unsigned int o = (unsigned int)offset;
    ms->getConstraintJacobian(cparams,J,o);
    offset = (int)o;
    return RESULT_CONTINUE;
}

} // namespace sofa::simulation::mechanicalvisitor

