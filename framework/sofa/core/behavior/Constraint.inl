/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINT_INL
#define SOFA_CORE_BEHAVIOR_CONSTRAINT_INL

#include <sofa/core/behavior/Constraint.h>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalState<DataTypes> *mm)
    : endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate(mm)
{
}

template<class DataTypes>
Constraint<DataTypes>::~Constraint()
{
}


template <class DataTypes>
bool Constraint<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}


template<class DataTypes>
void Constraint<DataTypes>::init()
{
    BaseConstraint::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}


template<class DataTypes>
void Constraint<DataTypes>::getConstraintViolation(defaulttype::BaseVector *v, const ConstraintParams* cParams)
{
    if (cParams)
    {
        getConstraintViolation(v, *cParams->readX(mstate), *cParams->readV(mstate), cParams);
    }
}


#ifndef SOFA_DEPRECATE_OLD_API
template<class DataTypes>
void Constraint<DataTypes>::getConstraintViolation(defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &/*v*/, const ConstraintParams* /*cParams*/)
{
    if (mstate)
    {
        bool freePos = false;

        if (&x.getValue() == mstate->getXfree())
            freePos = true;

        getConstraintValue(resV, freePos);
    }
}

template<class DataTypes>
void Constraint<DataTypes>::getConstraintValue(defaulttype::BaseVector * /*resV*/, bool /*freeMotion*/)
{
    serr << "ERROR(" << getClassName() << "): getConstraintViolation(defaulttype::BaseVector *, bool freeMotion) not implemented." << sendl;
}
#endif // SOFA_DEPRECATE_OLD_API


template<class DataTypes>
void Constraint<DataTypes>::buildConstraintMatrix(MultiMatrixDerivId cId, unsigned int &cIndex, const ConstraintParams* cParams)
{
    if (cParams)
    {
        buildConstraintMatrix(*cId[mstate].write(), cIndex, *cParams->readX(mstate), cParams);
    }
}


#ifndef SOFA_DEPRECATE_OLD_API
template<class DataTypes>
void Constraint<DataTypes>::buildConstraintMatrix(DataMatrixDeriv &/*c*/, unsigned int &/*cIndex*/, const DataVecCoord &/*x*/, const ConstraintParams* /*cParams*/)
{
    serr << "ERROR(" << getClassName()
            << "): buildConstraintMatrix(DataMatrixDeriv *c, unsigned int &cIndex, const DataVecCoord &x, const ConstraintParams* cParams) not implemented." << sendl;
}
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_CONSTRAINT_INL
