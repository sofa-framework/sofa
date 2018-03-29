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
#ifndef SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_INL
#define SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_INL

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <iostream>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::ProjectiveConstraintSet(MechanicalState<DataTypes> *mm)
    : endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate(initLink("mstate", "MechanicalState used by this projective constraint"), mm)
{
}

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::~ProjectiveConstraintSet()
{
}

template <class DataTypes>
bool ProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::init()
{
    BaseProjectiveConstraintSet::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    if(!mstate) this->serr<<"ProjectiveConstraintSet<DataTypes>::init(), no mstate . This may be because there is no MechanicalState in the local context, or because the type is not compatible." << this->sendl;
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId)
{
    if (!isActive())
        return;

    if (mstate)
    {
        projectJacobianMatrix(mparams, *cId[mstate.get(mparams)].write());
    }
}



template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId)
{
//    serr << "ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) " << this->getName() << sendl;

    if (!isActive())
        return;
    if (mstate)
    {
//        serr << "ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) " << this->getName() << " has mstate " << sendl;

            projectResponse(mparams, *dxId[mstate.get(mparams)].write());
    }
    else serr << "ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId), no mstate for " << this->getName() << sendl;
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId)
{
    if (!isActive())
        return;

    if (mstate)
    {

            projectVelocity(mparams, *vId[mstate.get(mparams)].write());
    }
    else serr << "ProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId dxId), no mstate for " << this->getName() << sendl;
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId)
{
    if (!isActive())
        return;

    if (mstate)
    {

            projectPosition(mparams, *xId[mstate.get(mparams)].write());
    }
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif
