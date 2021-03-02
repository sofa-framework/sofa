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
#include <sofa/helper/vector_T.h>

/// Specialization for reading vectors of int and unsigned int using "A-B" notation for all integers between A and B
template<> SOFA_HELPER_API std::istream& sofa::helper::vector<int>::read( std::istream& in );
template<> SOFA_HELPER_API std::istream& sofa::helper::vector<unsigned int>::read( std::istream& in );

/// Specialization for writing vectors of unsigned char
template<> SOFA_HELPER_API std::ostream& sofa::helper::vector<unsigned char>::write(std::ostream& os) const;
template<> SOFA_HELPER_API std::istream& sofa::helper::vector<unsigned char>::read(std::istream& in);

//#ifndef SOFA_HELPER_VECTOR_INTEGRAL_DEFINITION
//extern template class SOFA_HELPER_API sofa::helper::vector<bool>;
//extern template class SOFA_HELPER_API sofa::helper::vector<char>;
//extern template class SOFA_HELPER_API sofa::helper::vector<unsigned char>;
//extern template class SOFA_HELPER_API sofa::helper::vector<int>;
//extern template class SOFA_HELPER_API sofa::helper::vector<unsigned int>;
//extern template class SOFA_HELPER_API sofa::helper::vector<long>;
//extern template class SOFA_HELPER_API sofa::helper::vector<unsigned long>;
//extern template class SOFA_HELPER_API sofa::helper::vector<long long>;
//extern template class SOFA_HELPER_API sofa::helper::vector<unsigned long long>;
//#endif // SOFA_HELPER_VECTOR_INTEGRAL_DEFINITION
