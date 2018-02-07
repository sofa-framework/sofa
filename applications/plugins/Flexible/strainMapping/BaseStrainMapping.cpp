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
#include <sofa/core/Mapping.inl>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

namespace sofa
{
namespace core
{

using namespace defaulttype;

template class SOFA_Flexible_API Mapping< F331Types, E331Types >;
template class SOFA_Flexible_API Mapping< F321Types, E321Types >;
template class SOFA_Flexible_API Mapping< F311Types, E311Types >;
template class SOFA_Flexible_API Mapping< F332Types, E332Types >;
template class SOFA_Flexible_API Mapping< F221Types, E221Types >;
template class SOFA_Flexible_API Mapping< F332Types, E333Types >;

template class SOFA_Flexible_API Mapping< F331Types, I331Types >;

template class SOFA_Flexible_API Mapping< F331Types, U331Types >;
template class SOFA_Flexible_API Mapping< F321Types, U321Types >;

template class SOFA_Flexible_API Mapping< E331Types, E331Types >;
template class SOFA_Flexible_API Mapping< E321Types, E321Types >;
template class SOFA_Flexible_API Mapping< E311Types, E311Types >;
template class SOFA_Flexible_API Mapping< E332Types, E332Types >;
template class SOFA_Flexible_API Mapping< E333Types, E333Types >;


} // namespace core

} // namespace sofa

