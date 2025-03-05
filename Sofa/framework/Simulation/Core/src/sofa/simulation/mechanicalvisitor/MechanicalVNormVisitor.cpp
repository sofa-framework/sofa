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

#include <sofa/simulation/mechanicalvisitor/MechanicalVNormVisitor.h>

#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalVNormVisitor::fwdMechanicalState(VisitorContext* /*ctx*/, core::behavior::BaseMechanicalState* mm)
{
    if( l>0 ) accum += mm->vSum(this->params, a.getId(mm), l );
    else {
        const SReal mmax = mm->vMax(this->params, a.getId(mm) );
        if( mmax>accum ) accum=mmax;
    }
    return RESULT_CONTINUE;
}

SReal MechanicalVNormVisitor::getResult() const
{
    if( l>1 )
        return exp( log(accum) / l);
    else return accum;
}

std::string MechanicalVNormVisitor::getInfos() const
{
    std::string name("v= norm(a) with a[");
    name += a.getName() + "]";
    return name;
}

}