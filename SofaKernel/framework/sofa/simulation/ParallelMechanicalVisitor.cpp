/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/ParallelMechanicalVisitor.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{

Visitor::Result ParallelMechanicalVOpVisitor::fwdMechanicalState(Node* /*node*/, sofa::core::behavior::BaseMechanicalState* mm)
{

    mm->vOp(this->params /* PARAMS FIRST */, v.getId(mm), a.getId(mm), b.getId(mm), f, fSh);
    return RESULT_CONTINUE;
}
Visitor::Result ParallelMechanicalVOpVisitor::fwdMappedMechanicalState(Node* /*node*/, sofa::core::behavior::BaseMechanicalState* )
{
    //mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}


Visitor::Result ParallelMechanicalVOpMecVisitor::fwdMechanicalState(Node* /*node*/, sofa::core::behavior::BaseMechanicalState* mm)
{

    mm->vOpMEq(this->params /* PARAMS FIRST */, v.getId(mm), a.getId(mm), fSh);

    return RESULT_CONTINUE;
}



Visitor::Result ParallelMechanicalVOpMecVisitor::fwdMappedMechanicalState(Node* /*node*/, sofa::core::behavior::BaseMechanicalState* )
{
    //mm->vOp(v,a,b,f);
    return RESULT_CONTINUE;
}
Visitor::Result ParallelMechanicalVDotVisitor::fwdMechanicalState(Node* /*node*/, sofa::core::behavior::BaseMechanicalState* mm)
{
    mm->vDot(this->params /* PARAMS FIRST */, totalSh, a.getId(mm), b.getId(mm));
    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

