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
#define SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION
#include <sofa/linearalgebra/FullMatrix.inl>

namespace sofa::linearalgebra
{

#if defined(SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION)
std::ostream& operator<<(std::ostream& out, const FullMatrix<double>& v ){ return readFromStream(out, v); }
std::ostream& operator<<(std::ostream& out, const FullMatrix<float>& v ){ return readFromStream(out, v); }
template class FullMatrix<double>;
template class FullMatrix<float>;

std::ostream& operator<<(std::ostream& out, const LPtrFullMatrix<double>& v ){ return readFromStream(out, v); }
std::ostream& operator<<(std::ostream& out, const LPtrFullMatrix<float>& v ){ return readFromStream(out, v); }
template class LPtrFullMatrix<double>;
template class LPtrFullMatrix<float>;
#endif /// SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION

} /// namespace sofa::linearalgebra
