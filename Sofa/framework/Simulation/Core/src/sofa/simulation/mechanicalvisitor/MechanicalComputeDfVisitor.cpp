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

#include <sofa/simulation/mechanicalvisitor/MechanicalComputeDfVisitor.h>

#include <sofa/core/behavior/BaseForceField.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalComputeDfVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /* mm */)
{
    //<TO REMOVE>
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeDfVisitor::fwdMappedMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* /*mm*/)
{
    return RESULT_CONTINUE;
}

Visitor::Result MechanicalComputeDfVisitor::fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* ff)
{
    ff->addDForce(this->mparams, res);
    return RESULT_CONTINUE;
}

void MechanicalComputeDfVisitor::bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
{
    if (accumulate)
    {
        map->applyJT(mparams, res, res);  // apply material stiffness: variation of force below the mapping
        if( mparams->kFactor() ) map->applyDJT(mparams, res, res); // apply geometric stiffness: variation due to a change of mapping, with a constant force below the mapping
    }
}

void MechanicalComputeDfVisitor::bwdMechanicalState(simulation::Node* , core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(mm);
}

std::string MechanicalComputeDfVisitor::getInfos() const
{
    std::string name="["+res.getName()+"]";
    if (accumulate) name+= " Accumulating";
    else            name+= " Not Accumulating";
    return name;
}

} // namespace sofa::simulation::mechanicalvisitor
