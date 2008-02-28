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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MIXEDINTERACTIONCONSTRAINT_INL

#include <sofa/core/objectmodel/DataPtr.h>
#include "MixedInteractionConstraint.h"

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes1, class DataTypes2>
MixedInteractionConstraint<DataTypes1, DataTypes2>::MixedInteractionConstraint(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2)
    : endTime( initData(&endTime,(double)-1,"endTime","The constraint stops acting after the given value. Une a negative value for infinite constraints") )
    , mstate1(mm1), mstate2(mm2)
{
}

template<class DataTypes1, class DataTypes2>
MixedInteractionConstraint<DataTypes1, DataTypes2>::~MixedInteractionConstraint()
{
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::init()
{
    InteractionConstraint::init();
}

template<class DataTypes1, class DataTypes2>
bool MixedInteractionConstraint<DataTypes1, DataTypes2>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::projectResponse()
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
        projectResponse(*mstate1->getDx(), *mstate2->getDx());
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::projectVelocity()
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
        projectVelocity(*mstate1->getV(), *mstate2->getV());
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::projectPosition()
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
        projectPosition(*mstate1->getX(), *mstate2->getX());
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::applyConstraint(unsigned int &contactId)
{
    if( !isActive() ) return;
    if (mstate1 && mstate2)
        applyConstraint(*mstate1->getC(), *mstate2->getC(), contactId);
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
