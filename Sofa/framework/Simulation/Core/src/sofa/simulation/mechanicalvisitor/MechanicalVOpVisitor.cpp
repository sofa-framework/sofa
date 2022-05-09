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

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>

#include <sofa/core/BaseMapping.h>

namespace sofa::simulation::mechanicalvisitor
{

bool MechanicalVOpVisitor::stopAtMechanicalMapping(simulation::Node *, sofa::core::BaseMapping *map)
{
    if (mapped || only_mapped)
        return false;
    else
        return !map->areForcesMapped();
}

Visitor::Result MechanicalVOpVisitor::fwdMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    if (!only_mapped)
        mm->vOp(this->params, v.getId(mm) ,a.getId(mm),b.getId(mm),((ctx->nodeData && *ctx->nodeData != 1.0) ? *ctx->nodeData * f : f) );
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalVOpVisitor::fwdMappedMechanicalState(VisitorContext* ctx, core::behavior::BaseMechanicalState* mm)
{
    if (mapped || only_mapped)
    {
        mm->vOp(this->params, v.getId(mm) ,a.getId(mm),b.getId(mm),((ctx->nodeData && *ctx->nodeData != 1.0) ? *ctx->nodeData * f : f) );
    }
    return RESULT_CONTINUE;
}

std::string MechanicalVOpVisitor::getInfos() const
{
    std::string info="v=";
    std::string aLabel;
    std::string bLabel;
    std::string fLabel;

    std::ostringstream out;
    out << "f["<<f<<"]";
    fLabel+= out.str();

    if (!a.isNull())
    {
        info+="a";
        aLabel="a[" + a.getName() + "] ";
        if (!b.isNull())
        {
            info += "+b*f";
            bLabel += "b[" + b.getName() + "] ";
        }
    }
    else
    {
        if (!b.isNull())
        {
            info += "b*f";
            bLabel += "b[" + b.getName() + "] ";
        }
        else
        {
            info+="zero"; fLabel.clear();
        }
    }
    info += " : with v[" + v.getName() + "] " + aLabel + bLabel + fLabel;
    return info;
}

}