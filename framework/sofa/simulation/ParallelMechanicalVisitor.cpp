/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/common/ParallelMechanicalVisitor.h>
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

