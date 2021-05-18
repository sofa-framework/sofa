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
#define LMCONSTRAINT_MECHANICALWRITELMCONSTRAINT_CPP
#include <LMConstraint/MechanicalWriteLMConstraint.h>

namespace sofa
{

namespace simulation
{
using namespace sofa::core;


Visitor::Result MechanicalWriteLMConstraint::fwdConstraintSet(simulation::Node* /*node*/, core::behavior::BaseConstraintSet* c)
{
    if (core::behavior::BaseLMConstraint* LMc=dynamic_cast<core::behavior::BaseLMConstraint*>(c))
    {
        LMc->writeConstraintEquations(offset, id, order);
        datasC.push_back(LMc);
    }
    return RESULT_CONTINUE;
}

std::string MechanicalWriteLMConstraint::getInfos() const
{
    std::string name;
    if      (order == core::ConstraintParams::ACC)
        name= "["+sofa::core::VecId::dx().getName()+"]";
    else if (order == core::ConstraintParams::VEL)
        name= "["+sofa::core::VecId::velocity().getName()+"]";
    else if (order == core::ConstraintParams::POS)
        name= "["+sofa::core::VecId::position().getName()+"]";
    return name;
}



} // namespace simulation

} // namespace sofa

