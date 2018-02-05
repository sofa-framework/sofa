/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_INL
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONPROJECTIVECONSTRAINTSET_INL

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::PairInteractionProjectiveConstraintSet(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : endTime( initData(&endTime,(SReal)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate1(initLink("object1", "First object to constrain"), mm1)
    , mstate2(initLink("object2", "Second object to constrain"), mm2)
{
    if (!mm1)
        mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::~PairInteractionProjectiveConstraintSet()
{
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::init()
{
    BaseInteractionProjectiveConstraintSet::init();
    if (mstate1 == NULL || mstate2 == NULL)
    {
        mstate1 = mstate2 = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    }

    this->mask1 = &mstate1->forceMask;
    this->mask2 = &mstate2->forceMask;
}

template<class DataTypes>
bool PairInteractionProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(const MechanicalParams* /*mparams*/, MultiMatrixDerivId /*cId*/)
{
    serr << "NOT IMPLEMENTED YET" << sendl;
}



template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;

            projectResponse(mparams, *dxId[mstate1.get(mparams)].write(), *dxId[mstate2.get(mparams)].write());
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;

            projectVelocity(mparams, *vId[mstate1.get(mparams)].write(), *vId[mstate2.get(mparams)].write());
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
    {
        this->mask1 = &mstate1->forceMask;
        this->mask2 = &mstate2->forceMask;
            projectPosition(mparams, *xId[mstate1.get(mparams)].write(), *xId[mstate2.get(mparams)].write());
    }
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
