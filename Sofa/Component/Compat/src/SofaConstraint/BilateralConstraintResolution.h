/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h>

SOFA_DEPRECATED_HEADER("v22.06", "v23.06", "sofa/component/constraint/lagrangian/model/BilateralConstraintResolution.h")

namespace sofa::component::constraintset::bilateralconstraintresolution
{
    using BilateralConstraintResolution = sofa::component::constraint::lagrangian::model::BilateralConstraintResolution;
    using BilateralConstraintResolution3Dof = sofa::component::constraint::lagrangian::model::BilateralConstraintResolution3Dof;
    using BilateralConstraintResolutionNDof = sofa::component::constraint::lagrangian::model::BilateralConstraintResolutionNDof;
    
} // namespace sofa::component::constraintset::bilateralconstraintresolution

namespace sofa::component::constraintset
{
    using bilateralconstraintresolution::BilateralConstraintResolution;
    using bilateralconstraintresolution::BilateralConstraintResolution3Dof;
    using bilateralconstraintresolution::BilateralConstraintResolutionNDof;

} // namespace sofa::component::constraintset
