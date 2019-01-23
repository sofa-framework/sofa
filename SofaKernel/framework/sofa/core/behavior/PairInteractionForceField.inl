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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
PairInteractionForceField<DataTypes>::PairInteractionForceField(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : mstate1(initLink("object1", "First object in interaction"), mm1)
    , mstate2(initLink("object2", "Second object in interaction"), mm2)
{
    if (!mm1)
        mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
}


template<class DataTypes>
void PairInteractionForceField<DataTypes>::init()
{

    BaseInteractionForceField::init();

    if (mstate1.get() == NULL || mstate2.get() == NULL)
    {
        serr<< "Init of PairInteractionForceField " << getContext()->getName() << " failed!" << sendl;
        //getContext()->removeObject(this);
        return;
    }
}


template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* mparams, MultiVecDerivId fId )
{
    if (mstate1 && mstate2)
    {
            addForce( mparams, *fId[mstate1.get(mparams)].write()   , *fId[mstate2.get(mparams)].write()   ,
                    *mparams->readX(mstate1), *mparams->readX(mstate2),
                    *mparams->readV(mstate1), *mparams->readV(mstate2) );

        updateForceMask();
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId )
{
    if (mstate1 && mstate2)
    {
            addDForce(
                mparams, *dfId[mstate1.get(mparams)].write()    , *dfId[mstate2.get(mparams)].write()   ,
                *mparams->readDx(mstate1) , *mparams->readDx(mstate2) );
    }
    else
        serr<<"PairInteractionForceField<DataTypes>::addDForce(const MechanicalParams* /*mparams*/, MultiVecDerivId /*fId*/ ), mstate missing"<<sendl;
}

template<class DataTypes>
SReal PairInteractionForceField<DataTypes>::getPotentialEnergy(const MechanicalParams* mparams) const
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(mparams, *mparams->readX(mstate1),*mparams->readX(mstate2));
    else return 0.0;
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::updateForceMask()
{
    // the default implementation adds every dofs to the mask
    // this sould be overloaded by each forcefield to only add the implicated dofs subset to the mask
    mstate1->forceMask.assign( mstate1->getSize(), true );
    mstate2->forceMask.assign( mstate2->getSize(), true );
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
