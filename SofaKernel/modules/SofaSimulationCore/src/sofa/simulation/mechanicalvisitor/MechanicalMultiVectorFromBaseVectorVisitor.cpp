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
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorFromBaseVectorVisitor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
namespace sofa::simulation::mechanicalvisitor
{

MechanicalMultiVectorFromBaseVectorVisitor::MechanicalMultiVectorFromBaseVectorVisitor(
        const core::ExecParams* params, sofa::core::MultiVecId _dest,
        const linearalgebra::BaseVector * _src,
        const sofa::core::behavior::MultiMatrixAccessor* _matrix)
    : BaseMechanicalVisitor(params) , src(_src), dest(_dest), matrix(_matrix), offset(0)
{
}

MechanicalMultiVectorFromBaseVectorVisitor::Result MechanicalMultiVectorFromBaseVectorVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
{
    if (matrix) offset = matrix->getGlobalOffset(mm);
    if (src!= nullptr && offset >= 0)
    {
        unsigned int o = (unsigned int)offset;
        mm->copyFromBaseVector(dest.getId(mm), src, o);
        offset = (int)o;
    }

    return RESULT_CONTINUE;
}

} // namespace sofa::simulation::mechanicalvisitor

