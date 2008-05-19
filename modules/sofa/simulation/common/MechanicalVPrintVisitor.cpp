/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/common/MechanicalVPrintVisitor.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{


MechanicalVPrintVisitor::MechanicalVPrintVisitor( VecId v, std::ostream& out )
    : v_(v)
    , out_(out)
{}

Visitor::Result MechanicalVPrintVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        out_<<"[ ";
        (*node->mechanicalState).printDOF(v_,out_);
        out_<<"] ";
    }
    for_each(this, node, node->constraint, &MechanicalVPrintVisitor::fwdConstraint);
    return Visitor::RESULT_CONTINUE;
}
void MechanicalVPrintVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        out_<<"[ ";
        (*mm).printDOF(v_,out_);
        out_<<"] ";
    }
}





MechanicalVPrintWithElapsedTimeVisitor::MechanicalVPrintWithElapsedTimeVisitor( VecId v, unsigned time, std::ostream& out )
    : v_(v)
    ,count_(0)
    ,time_(time)
    , out_(out)
{}

Visitor::Result MechanicalVPrintWithElapsedTimeVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        count_+=(*node->mechanicalState).printDOFWithElapsedTime(v_,count_,time_,out_);
    }
    for_each(this, node, node->constraint, &MechanicalVPrintWithElapsedTimeVisitor::fwdConstraint);
    return Visitor::RESULT_CONTINUE;
}
void MechanicalVPrintWithElapsedTimeVisitor::fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
{
    core::componentmodel::behavior::BaseMechanicalState* mm = c->getDOFs();
    if (mm)
    {
        count_+=(*mm).printDOFWithElapsedTime(v_,count_,time_,out_);
    }
}


} // namespace simulation

} // namespace sofa

