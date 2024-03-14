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

#include <sofa/simulation/mechanicalvisitor/MechanicalGetMomentumVisitor.h>

#include <sofa/core/behavior/BaseMass.h>

namespace sofa::simulation::mechanicalvisitor
{

MechanicalGetMomentumVisitor::MechanicalGetMomentumVisitor(const core::MechanicalParams *mparams)
        : sofa::simulation::MechanicalVisitor(mparams)
{}

const type::Vec6 &MechanicalGetMomentumVisitor::getMomentum() const
{ return m_momenta; }

Visitor::Result MechanicalGetMomentumVisitor::fwdMass(simulation::Node *, core::behavior::BaseMass *mass)
{
    m_momenta += mass->getMomentum();
    return RESULT_CONTINUE;
}

void MechanicalGetMomentumVisitor::execute(sofa::core::objectmodel::BaseContext *c, bool precomputedTraversalOrder)
{
    m_momenta.clear();
    sofa::simulation::MechanicalVisitor::execute( c, precomputedTraversalOrder );
}
}