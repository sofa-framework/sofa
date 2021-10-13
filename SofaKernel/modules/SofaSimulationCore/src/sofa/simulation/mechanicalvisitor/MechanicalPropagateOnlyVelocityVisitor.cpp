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

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyVelocityVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

MechanicalPropagateOnlyVelocityVisitor::MechanicalPropagateOnlyVelocityVisitor(
        const sofa::core::MechanicalParams* mparams,
        SReal time, core::MultiVecDerivId v)
        : MechanicalVisitor(mparams), currentTime(time), v(v)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    setReadWriteVectors();
#endif
}

Visitor::Result MechanicalPropagateOnlyVelocityVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalPropagateOnlyVelocityVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    map->applyJ(mparams, v, v);

    return RESULT_CONTINUE;
}

void MechanicalPropagateOnlyVelocityVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(mm);
}

std::string MechanicalPropagateOnlyVelocityVisitor::getInfos() const
{ 
    std::string name="v["+v.getName()+"]"; 
    return name; 
}

} // namespace sofa::simulation::mechanicalvisitor
