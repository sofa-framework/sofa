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
#define SOFA_TYPE_QUAT_CPP

#include <sofa/type/Quat.inl>

namespace sofa::type
{

/// Explicit instanciation of the quaternions for double precision
template class SOFA_TYPE_API Quat<double>;
template SOFA_TYPE_API std::ostream& operator << ( std::ostream& out, const Quat<double>& v );
template SOFA_TYPE_API std::istream& operator >> ( std::istream& in, Quat<double>& v );

/// Explicit instanciation of the quaternions for single precision
template class SOFA_TYPE_API Quat<float>;
template SOFA_TYPE_API std::ostream& operator << ( std::ostream& out, const Quat<float>& v );
template SOFA_TYPE_API std::istream& operator >> ( std::istream& in, Quat<float>& v );

} /// namespace sofa::type
