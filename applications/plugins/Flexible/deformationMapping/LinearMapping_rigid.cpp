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
#define SOFA_COMPONENT_MAPPING_LINEARMAPPING_rigid_CPP

#include <Flexible/config.h>
#include "LinearMapping.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../types/AffineTypes.h"
#include "../types/QuadraticTypes.h"
#include "../types/DeformationGradientTypes.h"

#include "BaseDeformationMapping.inl"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(LinearMapping_rigid)

using namespace defaulttype;

// Register in the Factory
int LinearMappingClass_rigid = core::RegisterObject("Map child positions as a linear combination of parents.")
        .add< LinearMapping< Rigid3Types, Vec3Types > >()
        .add< LinearMapping< Rigid3Types, ExtVec3fTypes > >()
        .add< LinearMapping< Rigid3Types, F331Types > >()
        .add< LinearMapping< Rigid3Types, F321Types > >()
        .add< LinearMapping< Rigid3Types, F311Types > >()
        .add< LinearMapping< Rigid3Types, F332Types > >()
        .add< LinearMapping< Rigid3Types, Affine3Types > >()
        ;

template class SOFA_Flexible_API LinearMapping< Rigid3Types, Vec3Types >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, ExtVec3fTypes >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, F331Types >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, F321Types >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, F311Types >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, F332Types >;
template class SOFA_Flexible_API LinearMapping< Rigid3Types, Affine3Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

