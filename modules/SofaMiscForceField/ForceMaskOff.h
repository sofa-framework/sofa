/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_MAskCanceller_H
#define SOFA_COMPONENT_FORCEFIELD_MAskCanceller_H

#include "config.h"

#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace forcefield
{


/// hack to add every dofs to the force mask of the associated mstate
/// i.e. turn off the force mask for this branch
/// @author Matthieu Nesme
class ForceMaskOff: public sofa::core::behavior::BaseForceField
{
public:
    SOFA_CLASS(ForceMaskOff, BaseForceField);

    virtual void init()
    {
        BaseForceField::init();
        mstate = getContext()->getMechanicalState();
    }

    virtual void updateForceMask()
    {
        mstate->forceMask.assign( mstate->getSize(), true );
    }


    // other virtual functions do nothing
    virtual void addForce(const core::MechanicalParams*, core::MultiVecDerivId ) {}
    virtual void addDForce(const core::MechanicalParams*, core::MultiVecDerivId ) {}
    virtual SReal getPotentialEnergy( const core::MechanicalParams* = core::MechanicalParams::defaultInstance() ) const { return 0; }
    virtual void addKToMatrix(const core::MechanicalParams*, const core::behavior::MultiMatrixAccessor* ) {}

protected:

    core::behavior::BaseMechanicalState *mstate;
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_MAskCanceller_H
