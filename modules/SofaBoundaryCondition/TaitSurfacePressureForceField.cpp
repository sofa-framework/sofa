/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP
#include <SofaBoundaryCondition/TaitSurfacePressureForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TaitSurfacePressureForceField)

int TaitSurfacePressureForceFieldClass = core::RegisterObject("\
This component computes the volume enclosed by a surface mesh \
and apply a pressure force following Tait's equation: $P = P_0 - B((V/V_0)^\\gamma - 1)$.\n\
This ForceField can be used to apply :\n\
 * a constant pressure (set $B=0$ and use $P_0$)\n\
 * an ideal gas pressure (set $\\gamma=1$ and use $B$)\n\
 * a pressure from water (set $\\gamma=7$ and use $B$)")
#ifndef SOFA_FLOAT
        .add< TaitSurfacePressureForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TaitSurfacePressureForceField<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API TaitSurfacePressureForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API TaitSurfacePressureForceField<Vec3fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa
