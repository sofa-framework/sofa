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

#include <sofa/simulation/mechanicalvisitor/MechanicalVMultiOpVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalVMultiOpVisitor::fwdMechanicalState(VisitorContext* /*ctx*/, core::behavior::BaseMechanicalState* mm)
{
    mm->vMultiOp(this->params, ops );
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalVMultiOpVisitor::fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(ctx);
    if (mapped)
    {
        mm->vMultiOp(this->params, ops );
    }
    return RESULT_CONTINUE;
}


std::string MechanicalVMultiOpVisitor::getInfos() const
{
    std::ostringstream out;
    for(VMultiOp::const_iterator it = ops.begin(), itend = ops.end(); it != itend; ++it)
    {
        if (it != ops.begin())
            out << " ;   ";
        core::MultiVecId r = it->first;
        out << r.getName();
        const type::vector< std::pair< core::ConstMultiVecId, SReal > >& operands = it->second;
        const int nop = (int)operands.size();
        if (nop==0)
        {
            out << " = 0";
        }
        else if (nop==1)
        {
            if (operands[0].first.getName() == r.getName())
                out << " *= " << operands[0].second;
            else
            {
                out << " = " << operands[0].first.getName();
                if (operands[0].second != 1.0)
                    out << "*"<<operands[0].second;
            }
        }
        else
        {
            int i;
            if (operands[0].first.getName() == r.getName() && operands[0].second == 1.0)
            {
                out << " +=";
                i = 1;
            }
            else
            {
                out << " =";
                i = 0;
            }
            for (; i<nop; ++i)
            {
                out << " " << operands[i].first.getName();
                if (operands[i].second != 1.0)
                    out << "*"<<operands[i].second;
                if (i < nop-1)
                    out << " +";
            }
        }
    }
    return out.str();
}

}
