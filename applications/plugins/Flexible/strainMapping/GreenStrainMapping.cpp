/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_CPP

#include <Flexible/config.h>
#include "GreenStrainMapping.h"
#include <sofa/core/ObjectFactory.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(GreenStrainMapping)

using namespace defaulttype;

// Register in the Factory
int GreenStrainMappingClass = core::RegisterObject("Map Deformation Gradients to Green Lagrangian Strain (large deformations).")

        .add< GreenStrainMapping< F331Types, E331Types > >(true)
        .add< GreenStrainMapping< F321Types, E321Types > >()
        .add< GreenStrainMapping< F311Types, E311Types > >()
        .add< GreenStrainMapping< F332Types, E332Types > >()
        .add< GreenStrainMapping< F332Types, E333Types > >();

template class SOFA_Flexible_API GreenStrainMapping< F331Types, E331Types >;
template class SOFA_Flexible_API GreenStrainMapping< F321Types, E321Types >;
template class SOFA_Flexible_API GreenStrainMapping< F311Types, E311Types >;
template class SOFA_Flexible_API GreenStrainMapping< F332Types, E332Types >;
template class SOFA_Flexible_API GreenStrainMapping< F332Types, E333Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

