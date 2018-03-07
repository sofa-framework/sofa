/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/MechanicalVPrintVisitor.h>
#include <sofa/simulation/Node.h>

using namespace sofa::core;
namespace sofa
{

namespace simulation
{


MechanicalVPrintVisitor::MechanicalVPrintVisitor(const core::ExecParams* params, ConstMultiVecId v, std::ostream& out )
    : Visitor(params)
    , v_(v)
    , out_(out)
{
}

Visitor::Result MechanicalVPrintVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        ConstVecId id = v_.getId(node->mechanicalState);
        if (!id.isNull())
        {
            out_<<"[ ";
            (*node->mechanicalState).printDOF(id,out_);
            out_<<"] ";
        }
    }
    return Visitor::RESULT_CONTINUE;
}


MechanicalVPrintWithElapsedTimeVisitor::MechanicalVPrintWithElapsedTimeVisitor(const core::ExecParams* params, ConstMultiVecId vid, unsigned time, std::ostream& out )
    : Visitor (params)
    , v_(vid)
    , count_(0)
    , time_(time)
    , out_(out)
{
}

Visitor::Result MechanicalVPrintWithElapsedTimeVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() )
    {
        ConstVecId id = v_.getId(node->mechanicalState);
        if (!id.isNull())
        {
            count_+=(*node->mechanicalState).printDOFWithElapsedTime(id,count_,time_,out_);
        }
    }
    return Visitor::RESULT_CONTINUE;
}




DofPrintVisitor::DofPrintVisitor(const core::ExecParams* params, ConstMultiVecId v, const std::string& dofname, std::ostream& out )
    : Visitor(params)
    , v_(v)
    , out_(out)
    , dofname_(dofname)
{
}

Visitor::Result DofPrintVisitor::processNodeTopDown(simulation::Node* node)
{
    if( ! node->mechanicalState.empty() && node->mechanicalState->getName()==dofname_ )
    {
        ConstVecId id = v_.getId(node->mechanicalState);
        if (!id.isNull())
        {
            (*node->mechanicalState).printDOF(id,out_);
            out_<<" ";
        }
    }
    return Visitor::RESULT_CONTINUE;
}


} // namespace simulation

} // namespace sofa
