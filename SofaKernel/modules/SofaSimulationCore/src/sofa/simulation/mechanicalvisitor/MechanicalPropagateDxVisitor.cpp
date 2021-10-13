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

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateDxVisitor.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    //<TO REMOVE>
    return RESULT_CONTINUE;
}


Visitor::Result MechanicalPropagateDxVisitor::fwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    map->applyJ(mparams, dx, dx);

    return RESULT_CONTINUE;
}


void MechanicalPropagateDxVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(mm);
}

bool MechanicalPropagateDxVisitor::stopAtMechanicalMapping(simulation::Node *, sofa::core::BaseMapping *map)
{
    if (ignoreFlag)
        return false;
    else
        return !map->areForcesMapped();
}

std::string MechanicalPropagateDxVisitor::getInfos() const
{
    std::string name="["+dx.getName()+"]";
    return name;
}

} // namespace sofa::simulation::mechanicalvisitor
