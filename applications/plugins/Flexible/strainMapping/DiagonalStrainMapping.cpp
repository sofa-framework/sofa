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
#define SOFA_COMPONENT_MAPPING_DiagonalStrainMAPPING_CPP

#include "../initFlexible.h"
#include "DiagonalStrainMapping.h"
#include <sofa/core/ObjectFactory.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(DiagonalStrainMapping);

using namespace defaulttype;

// Register in the Factory
int DiagonalStrainMappingClass = core::RegisterObject("Map Deformation Gradients to Diagonalized Strain (small local deformations diagonalized with a SVD).")

        .add< DiagonalStrainMapping< F331Types, D331Types > >(true)
        .add< DiagonalStrainMapping< F321Types, D321Types > >()
        .add< DiagonalStrainMapping< F332Types, D332Types > >()

        .add< DiagonalStrainMapping< F331Types, E331Types > >()
        .add< DiagonalStrainMapping< F321Types, E321Types > >()
        .add< DiagonalStrainMapping< F332Types, E332Types > >()
        ;

template class SOFA_Flexible_API DiagonalStrainMapping< F331Types, D331Types >;
template class SOFA_Flexible_API DiagonalStrainMapping< F321Types, D321Types >;
template class SOFA_Flexible_API DiagonalStrainMapping< F332Types, D332Types >;

template class SOFA_Flexible_API DiagonalStrainMapping< F331Types, E331Types >;
template class SOFA_Flexible_API DiagonalStrainMapping< F321Types, E321Types >;
template class SOFA_Flexible_API DiagonalStrainMapping< F332Types, E332Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

