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
//
// C++ Implementation: MechanicalComputeEnergyVisitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/MechanicalComputeEnergyVisitor.h>

namespace sofa
{

namespace simulation
{


MechanicalComputeEnergyVisitor::MechanicalComputeEnergyVisitor(const core::MechanicalParams* mparams)
    : sofa::simulation::MechanicalVisitor(mparams)
    , m_kineticEnergy(0.)
    , m_potentialEnergy(0.)
{
}


MechanicalComputeEnergyVisitor::~MechanicalComputeEnergyVisitor()
{
}

SReal MechanicalComputeEnergyVisitor::getKineticEnergy()
{
    return m_kineticEnergy;
}

SReal MechanicalComputeEnergyVisitor::getPotentialEnergy()
{
    return m_potentialEnergy;
}


/// Process the BaseMass
Visitor::Result MechanicalComputeEnergyVisitor::fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass)
{
    m_kineticEnergy += (SReal)mass->getKineticEnergy();
    return RESULT_CONTINUE;
}
/// Process the BaseForceField
Visitor::Result MechanicalComputeEnergyVisitor::fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* f)
{
    m_potentialEnergy += (SReal)f->getPotentialEnergy();
    return RESULT_CONTINUE;
}

}

}
