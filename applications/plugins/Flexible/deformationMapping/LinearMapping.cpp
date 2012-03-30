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
#define SOFA_COMPONENT_MAPPING_LINEARMAPPING_CPP

#include "LinearMapping.inl"
#include <sofa/core/ObjectFactory.h>


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../frame/AffineTypes.h"
#include "../frame/QuadraticTypes.h"
#include "DeformationGradientTypes.h"

namespace sofa
{
namespace component
{

namespace mapping
{
SOFA_DECL_CLASS(LinearMapping);

using namespace defaulttype;

// Register in the Factory
int LinearMappingClass = core::RegisterObject("Map child positions as a linear combination of parents.")

#ifndef SOFA_FLOAT
        .add< LinearMapping< Vec3dTypes, Vec3dTypes > >(true)
        .add< LinearMapping< Vec3dTypes, ExtVec3fTypes > >()
        .add< LinearMapping< Vec3dTypes, DefGradient331dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearMapping< Vec3fTypes, Vec3fTypes > >()
        .add< LinearMapping< Vec3fTypes, ExtVec3fTypes > >()
        .add< LinearMapping< Vec3fTypes, DefGradient331fTypes > >()
#endif

        ;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API LinearMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_Flexible_API LinearMapping< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_Flexible_API LinearMapping< Vec3dTypes, DefGradient331dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API LinearMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_Flexible_API LinearMapping< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_Flexible_API LinearMapping< Vec3fTypes, DefGradient331fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

