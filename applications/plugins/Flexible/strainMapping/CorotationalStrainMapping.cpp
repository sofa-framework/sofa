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
#define SOFA_COMPONENT_MAPPING_CorotationalStrainMAPPING_CPP

#include <Flexible/config.h>
#include "CorotationalStrainMapping.h"
#include <sofa/core/ObjectFactory.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(CorotationalStrainMapping)

using namespace defaulttype;

// Register in the Factory
int CorotationalStrainMappingClass = core::RegisterObject("Map Deformation Gradients to Corotational Strain (small local deformations).")

        .add< CorotationalStrainMapping< F331Types, E331Types > >(true)
        .add< CorotationalStrainMapping< F321Types, E321Types > >()
        .add< CorotationalStrainMapping< F311Types, E311Types > >()
        .add< CorotationalStrainMapping< F332Types, E332Types > >()
        .add< CorotationalStrainMapping< F221Types, E221Types > >();

template class SOFA_Flexible_API CorotationalStrainMapping< F331Types, E331Types >;
template class SOFA_Flexible_API CorotationalStrainMapping< F321Types, E321Types >;
template class SOFA_Flexible_API CorotationalStrainMapping< F311Types, E311Types >;
template class SOFA_Flexible_API CorotationalStrainMapping< F332Types, E332Types >;
template class SOFA_Flexible_API CorotationalStrainMapping< F221Types, E221Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

