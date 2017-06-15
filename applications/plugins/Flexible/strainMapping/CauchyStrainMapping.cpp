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
#define SOFA_COMPONENT_MAPPING_CauchyStrainMAPPING_CPP

#include <Flexible/config.h>
#include "CauchyStrainMapping.h"
#include <sofa/core/ObjectFactory.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(CauchyStrainMapping)

using namespace defaulttype;

// Register in the Factory
int CauchyStrainMappingClass = core::RegisterObject("Map Deformation Gradients to Linear/Cauchy Strain (small displacements)")

        .add< CauchyStrainMapping< F331Types, E331Types > >(true)
        .add< CauchyStrainMapping< F321Types, E321Types > >()
        .add< CauchyStrainMapping< F311Types, E311Types > >()
        .add< CauchyStrainMapping< F332Types, E332Types > >();

template class SOFA_Flexible_API CauchyStrainMapping< F331Types, E331Types >;
template class SOFA_Flexible_API CauchyStrainMapping< F321Types, E321Types >;
template class SOFA_Flexible_API CauchyStrainMapping< F311Types, E311Types >;
template class SOFA_Flexible_API CauchyStrainMapping< F332Types, E332Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

