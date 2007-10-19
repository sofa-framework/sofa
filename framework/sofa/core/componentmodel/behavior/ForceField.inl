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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_INL
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_INL

#include <sofa/core/objectmodel/Field.h>
#include "ForceField.h"

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class DataTypes>
ForceField<DataTypes>::ForceField(MechanicalState<DataTypes> *mm)
    : mstate(mm)
{
}

template<class DataTypes>
ForceField<DataTypes>::~ForceField()
{
}

template<class DataTypes>
void ForceField<DataTypes>::init()
{
    BaseForceField::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}

template<class DataTypes>
void ForceField<DataTypes>::addForce()
{
    if (mstate)
        addForce(*mstate->getF(), *mstate->getX(), *mstate->getV());
}

template<class DataTypes>
void ForceField<DataTypes>::addDForce()
{
    if (mstate)
        addDForce(*mstate->getF(), *mstate->getDx());
}

template<class DataTypes>
void ForceField<DataTypes>::addDForceV()
{
    if (mstate)
        addDForce(*mstate->getF(), *mstate->getV());
}


template<class DataTypes>
double ForceField<DataTypes>::getPotentialEnergy()
{
    if (mstate)
        return getPotentialEnergy(*mstate->getX());
    else return 0;
}

template<class DataTypes>
void ForceField<DataTypes>::addKDxToVector(defaulttype::BaseVector *resVect, double kFact, unsigned int& offset)
{
    if (mstate)
    {
        addKDxToVector(resVect, mstate->getDx(), kFact, offset);
    }
}

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
