/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/interactionforcefield/LagrangeMultiplierInteraction.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;



SOFA_DECL_CLASS(LagrangeMultiplierInteraction)

int LagrangeMultiplierInteractionClass = core::RegisterObject("Lagrange Multiplier computation")
#ifndef SOFA_FLOAT
        .add< LagrangeMultiplierInteraction<Vec1dTypes, Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LagrangeMultiplierInteraction<Vec1fTypes, Vec3fTypes> >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< LagrangeMultiplierInteraction<Vec1dTypes, Vec3fTypes> >()
        .add< LagrangeMultiplierInteraction<Vec1fTypes, Vec3dTypes> >()
#endif
#endif
//.add< LagrangeMultiplierInteraction<Vec3fTypes, Rigid3fTypes> >()
//.add< LagrangeMultiplierInteraction<Vec3dTypes, Vec3dTypes> >()
//.add< LagrangeMultiplierInteraction<Vec3fTypes, Vec3fTypes> >()
        ;

#ifndef SOFA_FLOAT
template class LagrangeMultiplierInteraction<Vec1dTypes, Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class LagrangeMultiplierInteraction<Vec1fTypes, Vec3fTypes>;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class LagrangeMultiplierInteraction<Vec1dTypes, Vec3fTypes>;
template class LagrangeMultiplierInteraction<Vec1fTypes, Vec3dTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
