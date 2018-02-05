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
#define SOFA_COMPONENT_CONTAINER_SPATIALGRIDCONTAINER_CPP
#include <SofaSphFluid/SpatialGridContainer.inl>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;
using namespace core::behavior;


SOFA_DECL_CLASS(SpatialGridContainer)

int SpatialGridContainerClass = core::RegisterObject("Hashing spatial grid container, used for SPH fluids for instance.")
#ifndef SOFA_FLOAT
        .add< SpatialGridContainer<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SpatialGridContainer<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SpatialGridContainer< Vec3dTypes >;
template class SOFA_SPH_FLUID_API SpatialGrid< SpatialGridTypes< Vec3dTypes > >;
#endif
#ifndef SOFA_DOUBLE
template class SpatialGridContainer< Vec3fTypes >;
template class SOFA_SPH_FLUID_API SpatialGrid< SpatialGridTypes< Vec3fTypes > >;
#endif

} // namespace container

} // namespace component

} // namespace sofa

