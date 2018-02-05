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
#define FLEXIBLE_StrainTYPES_CPP

#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/State.inl>
#include <SofaBaseMechanics/MechanicalObject.inl>

namespace sofa
{
namespace component
{
namespace container
{

// ==========================================================================
// Instanciation

SOFA_DECL_CLASS ( StrainMechanicalObject )

using namespace sofa::defaulttype;

int StrainMechanicalObjectClass = core::RegisterObject ( "mechanical state vectors" )
#ifndef SOFA_FLOAT
        .add< MechanicalObject<E331dTypes> >()
        .add< MechanicalObject<E321dTypes> >()
        .add< MechanicalObject<E311dTypes> >()
        .add< MechanicalObject<E332dTypes> >()
        .add< MechanicalObject<E333dTypes> >()
        .add< MechanicalObject<E221dTypes> >()

//        .add< MechanicalObject<D331dTypes> >()
//        .add< MechanicalObject<D321dTypes> >()
//        .add< MechanicalObject<D332dTypes> >()
//        .add< MechanicalObject<D333dTypes> >()

        .add< MechanicalObject<I331dTypes> >()
//.add< MechanicalObject<I332dTypes> >()
//.add< MechanicalObject<I333dTypes> >()

        .add< MechanicalObject<U331dTypes> >()
        .add< MechanicalObject<U321dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< MechanicalObject<E331fTypes> >()
        .add< MechanicalObject<E321fTypes> >()
        .add< MechanicalObject<E311fTypes> >()
        .add< MechanicalObject<E332fTypes> >()
        .add< MechanicalObject<E333fTypes> >()
        .add< MechanicalObject<E221fTypes> >()

//        .add< MechanicalObject<D331fTypes> >()
//        .add< MechanicalObject<D321fTypes> >()
//        .add< MechanicalObject<D332fTypes> >()
//        .add< MechanicalObject<D333fTypes> >()

        .add< MechanicalObject<I331fTypes> >()
//.add< MechanicalObject<I332fTypes> >()
//.add< MechanicalObject<I333fTypes> >()

        .add< MechanicalObject<U331fTypes> >()
        .add< MechanicalObject<U321fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API MechanicalObject<E331dTypes>;
template class SOFA_Flexible_API MechanicalObject<E321dTypes>;
template class SOFA_Flexible_API MechanicalObject<E311dTypes>;
template class SOFA_Flexible_API MechanicalObject<E332dTypes>;
template class SOFA_Flexible_API MechanicalObject<E333dTypes>;
template class SOFA_Flexible_API MechanicalObject<E221dTypes>;

//template class SOFA_Flexible_API MechanicalObject<D331dTypes>;
//template class SOFA_Flexible_API MechanicalObject<D321dTypes>;
//template class SOFA_Flexible_API MechanicalObject<D332dTypes>;
//template class SOFA_Flexible_API MechanicalObject<D333dTypes>;

template class SOFA_Flexible_API MechanicalObject<I331dTypes>;
//template class SOFA_Flexible_API MechanicalObject<I332dTypes>;
//template class SOFA_Flexible_API MechanicalObject<I333dTypes>;

template class SOFA_Flexible_API MechanicalObject<U331dTypes>;
template class SOFA_Flexible_API MechanicalObject<U321dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API MechanicalObject<E331fTypes>;
template class SOFA_Flexible_API MechanicalObject<E321fTypes>;
template class SOFA_Flexible_API MechanicalObject<E311fTypes>;
template class SOFA_Flexible_API MechanicalObject<E332fTypes>;
template class SOFA_Flexible_API MechanicalObject<E333fTypes>;
template class SOFA_Flexible_API MechanicalObject<E221fTypes>;

//template class SOFA_Flexible_API MechanicalObject<D331fTypes>;
//template class SOFA_Flexible_API MechanicalObject<D321fTypes>;
//template class SOFA_Flexible_API MechanicalObject<D332fTypes>;
//template class SOFA_Flexible_API MechanicalObject<D333fTypes>;

template class SOFA_Flexible_API MechanicalObject<I331fTypes>;
//template class SOFA_Flexible_API MechanicalObject<I332fTypes>;
//template class SOFA_Flexible_API MechanicalObject<I333fTypes>;

template class SOFA_Flexible_API MechanicalObject<U331fTypes>;
template class SOFA_Flexible_API MechanicalObject<U321fTypes>;
#endif

} // namespace container
} // namespace component
} // namespace sofa
