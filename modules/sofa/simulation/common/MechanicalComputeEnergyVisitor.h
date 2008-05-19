/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Interface: MechanicalComputeEnergyVisitor
//
// Description:
//
//
// Author: Francois Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_MECHANICALCOMPUTEENERGYACTION_H
#define SOFA_SIMULATION_MECHANICALCOMPUTEENERGYACTION_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace simulation
{

/**
Compute the amount of mechanical energy

	@author Francois Faure
*/
class MechanicalComputeEnergyVisitor : public sofa::simulation::MechanicalVisitor
{
    SReal m_kineticEnergy;
    SReal m_potentialEnergy;

public:
    MechanicalComputeEnergyVisitor();

    ~MechanicalComputeEnergyVisitor();

    double getKineticEnergy();

    double getPotentialEnergy();

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::componentmodel::behavior::BaseMass* mass);

    /// Process the BaseForceField
    virtual Result fwdForceField(simulation::Node* /*node*/, core::componentmodel::behavior::BaseForceField* f);

};

}

}

#endif
