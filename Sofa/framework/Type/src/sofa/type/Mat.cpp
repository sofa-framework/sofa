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
#include <sofa/type/Mat.h>

namespace sofa::type
{
template class SOFA_TYPE_API Mat<2, 2, SReal>;
template class SOFA_TYPE_API Mat<2, 3, SReal>;
template class SOFA_TYPE_API Mat<3, 3, SReal>;
template class SOFA_TYPE_API Mat<4, 4, SReal>;
template class SOFA_TYPE_API Mat<6, 3, SReal>;
template class SOFA_TYPE_API Mat<6, 6, SReal>;
template class SOFA_TYPE_API Mat<8, 3, SReal>;
template class SOFA_TYPE_API Mat<8, 8, SReal>;
template class SOFA_TYPE_API Mat<9, 9, SReal>;
template class SOFA_TYPE_API Mat<12, 3, SReal>;
template class SOFA_TYPE_API Mat<12, 6, SReal>;
template class SOFA_TYPE_API Mat<12, 12, SReal>;
template class SOFA_TYPE_API Mat<20, 20, SReal>;
template class SOFA_TYPE_API Mat<20, 32, SReal>;
template class SOFA_TYPE_API Mat<24, 24, SReal>;
template class SOFA_TYPE_API Mat<32, 20, SReal>;
}
