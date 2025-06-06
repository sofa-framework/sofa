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

#include <string_view>

namespace sofa::core
{

/// Description of the constraint order.
///
/// The order corresponds to the derivative order of the constraint function. This information
/// tells which derivative is solved by the constraint solver. For example, solving only the
/// velocity-level will authorize constraint violation but will prevent further violation.
enum class ConstraintOrder
{
    POS = 0, //corresponds to the constraint function itself
    VEL, //corresponds to the first derivative of the constraint function
    ACC, //corresponds to the second derivative of the constraint function
    POS_AND_VEL
};

constexpr static std::string_view constOrderToString(ConstraintOrder order)
{
    if (order == sofa::core::ConstraintOrder::POS)
    {
        return "POSITION";
    }
    if (order == sofa::core::ConstraintOrder::VEL)
    {
        return "VELOCITY";
    }
    if (order == sofa::core::ConstraintOrder::ACC)
    {
        return "ACCELERATION";
    }
    if (order == sofa::core::ConstraintOrder::POS_AND_VEL)
    {
        return "POSITION AND VELOCITY";
    }
    return {};
}

}
