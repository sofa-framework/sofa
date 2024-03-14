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
#pragma once

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa::core::behavior
{

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::MixedInteractionForceField(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2)
    : Inherit1(), Inherit2(mm1, mm2)
{
    if (!mm1)
        this->mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        this->mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes1, class DataTypes2>
MixedInteractionForceField<DataTypes1, DataTypes2>::~MixedInteractionForceField()
{
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    
    if (this->mstate1 && this->mstate2)
    {
        auto state1 = this->mstate1.get();
        auto state2 = this->mstate2.get();
        addForce( mparams, *fId[state1].write(), *fId[state2].write(),
                  *mparams->readX(state1), *mparams->readX(state2),
                  *mparams->readV(state1), *mparams->readV(state2));
        
    }
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionForceField<DataTypes1, DataTypes2>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (this->mstate1 && this->mstate2)
    {
        auto state1 = this->mstate1.get();
        auto state2 = this->mstate2.get();
        addDForce( mparams, 
                   *dfId[state1].write()    , *dfId[state2].write()   ,
                   *mparams->readDx(state1) , *mparams->readDx(state2) );
    }
}



template<class DataTypes1, class DataTypes2>
SReal MixedInteractionForceField<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (this->mstate1 && this->mstate2)
        return getPotentialEnergy(mparams, *mparams->readX(this->mstate1.get()),*mparams->readX(this->mstate2.get()));
    else return 0;
}

} // namespace sofa::core::behavior

