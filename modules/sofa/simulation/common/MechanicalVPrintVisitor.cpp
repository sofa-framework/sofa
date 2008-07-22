/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/MechanicalVPrintVisitor.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{

MechanicalVPrintVisitor::MechanicalVPrintVisitor( VecId v, std::ostream& out )
    : v_(v)
    , out_(out)
{
}

Visitor::Result MechanicalVPrintVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        out_<<"[ ";
        (*node->mechanicalState).printDOF(v_,out_);
        out_<<"] ";
    }
    return Visitor::RESULT_CONTINUE;
}


MechanicalVPrintWithElapsedTimeVisitor::MechanicalVPrintWithElapsedTimeVisitor( VecId v, unsigned time, std::ostream& out )
    : v_(v)
    , count_(0)
    , time_(time)
    , out_(out)
{
}

Visitor::Result MechanicalVPrintWithElapsedTimeVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        count_+=(*node->mechanicalState).printDOFWithElapsedTime(v_,count_,time_,out_);
    }
    return Visitor::RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa
