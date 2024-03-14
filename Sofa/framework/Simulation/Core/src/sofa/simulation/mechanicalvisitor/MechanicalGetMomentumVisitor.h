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
#ifndef SOFA_SIMULATION_MECHANICALGETMOMENTUMVISITOR_H
#define SOFA_SIMULATION_MECHANICALGETMOMENTUMVISITOR_H

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/type/Vec.h>

namespace sofa::simulation::mechanicalvisitor
{

/// Compute the linear and angular momenta
///
/// @author Matthieu Nesme, 2015
///
class SOFA_SIMULATION_CORE_API MechanicalGetMomentumVisitor : public sofa::simulation::MechanicalVisitor
{
    type::Vec6 m_momenta;

public:
    MechanicalGetMomentumVisitor(const core::MechanicalParams* mparams);

    const type::Vec6& getMomentum() const;

    /// Process the BaseMass
    virtual Result fwdMass(simulation::Node* /*node*/, core::behavior::BaseMass* mass);


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalGetMomentumVisitor"; }

    virtual void execute( sofa::core::objectmodel::BaseContext* c, bool precomputedTraversalOrder=false );


};

}

#endif
