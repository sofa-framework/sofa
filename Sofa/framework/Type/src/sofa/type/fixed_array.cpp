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
#define FIXED_ARRAY_CPP

#include <sofa/type/fixed_array.h>

namespace sofa::type
{

template class SOFA_TYPE_API fixed_array<float, 2>;
template class SOFA_TYPE_API fixed_array<double, 2>;

template class SOFA_TYPE_API fixed_array<float, 3>;
template class SOFA_TYPE_API fixed_array<double, 3>;

template class SOFA_TYPE_API fixed_array<float, 4>;
template class SOFA_TYPE_API fixed_array<double, 4>;

template class SOFA_TYPE_API fixed_array<float, 5>;
template class SOFA_TYPE_API fixed_array<double, 5>;

template class SOFA_TYPE_API fixed_array<float, 6>;
template class SOFA_TYPE_API fixed_array<double, 6>;

template class SOFA_TYPE_API fixed_array<float, 7>;
template class SOFA_TYPE_API fixed_array<double, 7>;

} // namespace sofa::type

