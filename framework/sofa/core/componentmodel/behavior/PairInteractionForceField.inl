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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_PAIRINTERACTIONFORCEFIELD_INL

#include <sofa/core/objectmodel/DataPtr.h>
#include "PairInteractionForceField.h"

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
PairInteractionForceField<DataTypes>::PairInteractionForceField(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : mstate1(mm1), mstate2(mm2)
{
}

template<class DataTypes>
PairInteractionForceField<DataTypes>::~PairInteractionForceField()
{
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::init()
{
    InteractionForceField::init();
    if (mstate1 == NULL || mstate2 == NULL)
    {
        mstate1 = mstate2 = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    }
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addForce()
{
    if (mstate1 && mstate2)
        addForce(*mstate1->getF(), *mstate2->getF(),
                *mstate1->getX(), *mstate2->getX(),
                *mstate1->getV(), *mstate2->getV());
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForce()
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getDx(), *mstate2->getDx());
}

template<class DataTypes>
void PairInteractionForceField<DataTypes>::addDForceV()
{
    if (mstate1 && mstate2)
        addDForce(*mstate1->getF(),  *mstate2->getF(),
                *mstate1->getV(), *mstate2->getV());
}


template<class DataTypes>
double PairInteractionForceField<DataTypes>::getPotentialEnergy()
{
    if (mstate1 && mstate2)
        return getPotentialEnergy(*mstate1->getX(), *mstate2->getX());
    else return 0;
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
