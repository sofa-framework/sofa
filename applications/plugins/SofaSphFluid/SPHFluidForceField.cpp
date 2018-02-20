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
#define SOFA_COMPONENT_FORCEFIELD_SPHFLUIDFORCEFIELD_CPP

#include <SofaSphFluid/SPHFluidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::behavior;


SOFA_DECL_CLASS(SPHFluidForceField)

int SPHFluidForceFieldClass = core::RegisterObject("Smooth Particle Hydrodynamics")
#ifndef SOFA_FLOAT
        .add< SPHFluidForceField<Vec3dTypes> >()
//.add< SPHFluidForceField<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SPHFluidForceField<Vec3fTypes> >()
//.add< SPHFluidForceField<Vec2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3dTypes>;
template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec3fTypes>;
template class SOFA_SPH_FLUID_API SPHFluidForceField<Vec2fTypes>;
#endif


} // namespace forcefield

} // namespace component

} // namespace sofa

