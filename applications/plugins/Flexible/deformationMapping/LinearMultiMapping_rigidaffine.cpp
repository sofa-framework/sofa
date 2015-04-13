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
#define SOFA_COMPONENT_MAPPING_LINEARMULTIMAPPING_rigidaffine_CPP

#include "../initFlexible.h"
#include "LinearMultiMapping.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../types/AffineTypes.h"
#include "../types/QuadraticTypes.h"
#include "../types/DeformationGradientTypes.h"


#include "BaseDeformationMultiMapping.inl"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(LinearMultiMapping_rigidaffine)

using namespace defaulttype;

template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, Vec3Types >, defaulttype::LinearJacobianBlock< Affine3Types, Vec3Types > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, ExtVec3fTypes>,  defaulttype::LinearJacobianBlock< Affine3Types, ExtVec3fTypes > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, F331Types >, defaulttype::LinearJacobianBlock< Affine3Types, F331Types > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, F321Types >, defaulttype::LinearJacobianBlock< Affine3Types, F321Types > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, F311Types >, defaulttype::LinearJacobianBlock< Affine3Types, F311Types > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, F332Types >, defaulttype::LinearJacobianBlock< Affine3Types, F332Types > >;
template class SOFA_Flexible_API BaseDeformationMultiMappingT< defaulttype::LinearJacobianBlock< Rigid3Types, Affine3Types >, defaulttype::LinearJacobianBlock< Affine3Types, Affine3Types > >;


// Register in the Factory
int LinearMultiMappingClass_rigid = core::RegisterObject("Map child positions as a linear combination of parents.")
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, Vec3Types > >(true)
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, ExtVec3fTypes > >()
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, F331Types > >()
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, F321Types > >()
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, F311Types > >()
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, F332Types > >()
        .add< LinearMultiMapping< Rigid3Types, Affine3Types, Affine3Types > >()
        ;

template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, Vec3Types >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, ExtVec3fTypes >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, F331Types >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, F321Types >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, F311Types >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, F332Types >;
template class SOFA_Flexible_API LinearMultiMapping< Rigid3Types, Affine3Types, Affine3Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

