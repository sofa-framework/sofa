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
#define SOFA_COMPONENT_MAPPING_InvariantMAPPING_CPP

#include <Flexible/config.h>
#include "../strainMapping/InvariantMapping.h"
#include <sofa/core/ObjectFactory.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace component
{
namespace mapping
{

SOFA_DECL_CLASS(InvariantMapping)

using namespace defaulttype;

// Register in the Factory
int InvariantMappingClass = core::RegisterObject("Map deformation gradients to the invariants of the right Cauchy Green deformation tensor: I1, I2 and J")

        .add< InvariantMapping< F331Types, I331Types > >(true)
//.add< InvariantMapping< F332Types, I332Types > >()
//.add< InvariantMapping< F332Types, I333Types > >()

//        .add< InvariantMapping< U331Types, I331Types > >()
        ;

template class SOFA_Flexible_API InvariantMapping< F331Types, I331Types >;
//template class SOFA_Flexible_API InvariantMapping< F332Types, I332Types >;
//template class SOFA_Flexible_API InvariantMapping< F332Types, I333Types >;
//template class SOFA_Flexible_API InvariantMapping< U331Types, I331Types >;

} // namespace mapping
} // namespace component
} // namespace sofa

