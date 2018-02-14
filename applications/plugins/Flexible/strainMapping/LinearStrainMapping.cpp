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
#define SOFA_COMPONENT_LINEARSTRAINMAPPING_CPP

#include "LinearStrainMapping.h"

#include <sofa/core/ObjectFactory.h>

#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(LinearStrainMapping)

using namespace defaulttype;

// Register in the Factory
int LinearStrainMappingClass = core::RegisterObject("Map strain positions as a linear combination of strains, for smoothing.")

        .add< LinearStrainMapping< E331Types > >(true)
        .add< LinearStrainMapping< E311Types > >()
        .add< LinearStrainMapping< E321Types > >()
        .add< LinearStrainMapping< E332Types > >()
        .add< LinearStrainMapping< E333Types > >()

//        .add< LinearStrainMapping< D331Types > >()
//        .add< LinearStrainMapping< D321Types > >()
//        .add< LinearStrainMapping< D332Types > >()
        ;

template class SOFA_Flexible_API LinearStrainMapping< E331Types >;
template class SOFA_Flexible_API LinearStrainMapping< E311Types >;
template class SOFA_Flexible_API LinearStrainMapping< E321Types >;
template class SOFA_Flexible_API LinearStrainMapping< E332Types >;
template class SOFA_Flexible_API LinearStrainMapping< E333Types >;

//template class SOFA_Flexible_API LinearStrainMapping< D331Types >;
//template class SOFA_Flexible_API LinearStrainMapping< D321Types >;
//template class SOFA_Flexible_API LinearStrainMapping< D332Types >;


} // namespace mapping
} // namespace component
} // namespace sofa
