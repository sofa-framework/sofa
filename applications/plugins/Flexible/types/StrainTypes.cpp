/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define FLEXIBLE_StrainTYPES_CPP

#include "../initFlexible.h"
#include "../types/StrainTypes.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MechanicalObject.inl>

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

        .add< MechanicalObject<E331Types> >()
        .add< MechanicalObject<E321Types> >()
        .add< MechanicalObject<E311Types> >()
        .add< MechanicalObject<E332Types> >()
        .add< MechanicalObject<E333Types> >()

        .add< MechanicalObject<D331Types> >()
        .add< MechanicalObject<D321Types> >()
        .add< MechanicalObject<D332Types> >()
        .add< MechanicalObject<D333Types> >()

        .add< MechanicalObject<I331Types> >()
//.add< MechanicalObject<I332Types> >()
//.add< MechanicalObject<I333Types> >()

        .add< MechanicalObject<U331Types> >()
        .add< MechanicalObject<U321Types> >()
        ;

template class SOFA_Flexible_API MechanicalObject<E331Types>;
template class SOFA_Flexible_API MechanicalObject<E321Types>;
template class SOFA_Flexible_API MechanicalObject<E311Types>;
template class SOFA_Flexible_API MechanicalObject<E332Types>;
template class SOFA_Flexible_API MechanicalObject<E333Types>;

template class SOFA_Flexible_API MechanicalObject<D331Types>;
template class SOFA_Flexible_API MechanicalObject<D321Types>;
template class SOFA_Flexible_API MechanicalObject<D332Types>;
template class SOFA_Flexible_API MechanicalObject<D333Types>;

template class SOFA_Flexible_API MechanicalObject<I331Types>;
//template class SOFA_Flexible_API MechanicalObject<I332Types>;
//template class SOFA_Flexible_API MechanicalObject<I333Types>;

template class SOFA_Flexible_API MechanicalObject<U331Types>;
template class SOFA_Flexible_API MechanicalObject<U321Types>;

} // namespace container
} // namespace component
} // namespace sofa
