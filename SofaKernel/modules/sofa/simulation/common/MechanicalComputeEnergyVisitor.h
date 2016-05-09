/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
#include <sofa/core/MechanicalParams.h>
namespace sofa
{

namespace simulation
{

/**
Compute the amount of mechanical energy

	@author Francois Faure
*/
class SOFA_SIMULATION_COMMON_API MechanicalComputeEnergyVisitor : public sofa::simulation::MechanicalVisitor
{
    SReal m_kineticEnergy;
    SReal m_potentialEnergy;

public:
    MechanicalComputeEnergyVisitor(const core::MechanicalParams* mparams);

    ~MechanicalComputeEnergyVisitor();

    SReal getKineticEnergy();

    SReal getPotentialEnergy();

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass);

    /// Process the BaseForceField
    virtual Result fwdForceField(simulation::Node* /*node*/, core::behavior::BaseForceField* f);

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalComputeEnergyVisitor"; }

    virtual void execute( sofa::core::objectmodel::BaseContext* c, bool precomputedTraversalOrder=false )
    {
        m_kineticEnergy = m_potentialEnergy = 0;
        sofa::simulation::MechanicalVisitor::execute( c, precomputedTraversalOrder );
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void setReadWriteVectors()
    {
    }
#endif

};

}

}

#endif
