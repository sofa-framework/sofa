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
#include <SofaBaseLinearSolver/FullMatrix.inl>

namespace sofa::component::linearsolver
{

#if defined(SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION)
template class FullMatrix<double>;
template class FullMatrix<float>;
template class FullMatrix<bool>;

std::ostream& operator<<(std::ostream& out, const FullMatrix<double>& v ){ return readFromStream(out, v); }
std::ostream& operator<<(std::ostream& out, const FullMatrix<float>& v ){ return readFromStream(out, v); }
std::ostream& operator<<(std::ostream& out, const FullMatrix<bool>& v ){ return readFromStream(out, v); }

template class LPtrFullMatrix<double>;
template class LPtrFullMatrix<float>;
template class LPtrFullMatrix<bool>;
#endif /// SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION

} /// namespace sofa::component::linearsolver
