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
#define SOFA_TYPE_MAT_CPP

#include <sofa/type/Mat.h>

namespace sofa::type
{

template class SOFA_TYPE_API Mat<2,2,float>;
template class SOFA_TYPE_API Mat<2,2,double>;

template class SOFA_TYPE_API Mat<3,3,float>;
template class SOFA_TYPE_API Mat<3,3,double>;

template class SOFA_TYPE_API Mat<4,4,float>;
template class SOFA_TYPE_API Mat<4,4,double>;

template class SOFA_TYPE_API Mat<6,6,float>;
template class SOFA_TYPE_API Mat<6,6,double>;

template class SOFA_TYPE_API Mat<12,12,float>;
template class SOFA_TYPE_API Mat<12,12,double>;


} // namespace sofa::type
