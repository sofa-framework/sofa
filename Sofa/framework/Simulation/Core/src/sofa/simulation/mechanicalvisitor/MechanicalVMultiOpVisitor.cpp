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
    if (mapped)
    {
        if (ctx->nodeData && *ctx->nodeData != 1.0)
        {
            VMultiOp ops2 = ops;
            const SReal fact = *ctx->nodeData;
            for (auto& op : ops2)
            {
                auto& linearCombination = op.getLinearCombination();
                for (std::size_t i = 1; i < op.getLinearCombination().size(); ++i)
                {
                    linearCombination[i].factor *= fact;
                }
            }
            mm->vMultiOp(this->params, ops2 );
        }
        else
        {
            mm->vMultiOp(this->params, ops );
        }
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
        core::MultiVecId r = it->getOutput();
        out << r.getName();
        const auto& operands = it->getLinearCombination();
        const std::size_t nop = operands.size();
        if (nop==0)
        {
            out << " = 0";
        }
        else if (nop==1)
        {
            if (operands[0].id.getName() == r.getName())
                out << " *= " << operands[0].factor;
            else
            {
                out << " = " << operands[0].id.getName();
                if (operands[0].factor != 1.0)
                    out << "*"<<operands[0].factor;
            }
        }
        else
        {
            int i;
            if (operands[0].id.getName() == r.getName() && operands[0].factor == 1.0)
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
                out << " " << operands[i].id.getName();
                if (operands[i].factor != 1.0)
                    out << "*"<<operands[i].factor;
                if (i < nop-1)
                    out << " +";
            }
        }
    }
    return out.str();
}

}
