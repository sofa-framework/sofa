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
#define SOFA_COMPONENT_MAPPING_GreenStrainMAPPING_CPP

#include "GreenStrainMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include "DeformationGradientTypes.h"
#include "StrainTypes.h"

namespace sofa
{
namespace component
{

namespace mapping
{
SOFA_DECL_CLASS(GreenStrainMapping);

using namespace defaulttype;

// Register in the Factory
int GreenStrainMappingClass = core::RegisterObject("Map Deformation Gradients to Green Lagrangian Strain (large deformations).")

#ifndef SOFA_FLOAT
        .add< GreenStrainMapping< DefGradient331dTypes, Strain331dTypes > >(true)
//.add< GreenStrainMapping< DefGradient332dTypes, Strain332dTypes > >()
//.add< GreenStrainMapping< DefGradient332dTypes, Strain333dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< GreenStrainMapping< DefGradient331fTypes, Strain331fTypes > >()
//.add< GreenStrainMapping< DefGradient332fTypes, Strain332fTypes > >()
//.add< GreenStrainMapping< DefGradient332fTypes, Strain333fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_Flexible_API GreenStrainMapping< DefGradient331dTypes, Strain331dTypes >;
//template class SOFA_Flexible_API GreenStrainMapping< DefGradient332dTypes, Strain332dTypes >;
//template class SOFA_Flexible_API GreenStrainMapping< DefGradient332dTypes, Strain333dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API GreenStrainMapping< DefGradient331fTypes, Strain331fTypes >;
//template class SOFA_Flexible_API GreenStrainMapping< DefGradient332fTypes, Strain332fTypes >;
//template class SOFA_Flexible_API GreenStrainMapping< DefGradient332fTypes, Strain333fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

