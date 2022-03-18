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
* Authors: The SOFA Team && external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFABASELINEARSOLVER_EIGEN_EIGENVECTOR_CPP
#include <sofa/linearalgebra/eigen/EigenVector.h>

namespace sofa::linearalgebra::eigen
{

std::ostream& operator<<(std::ostream& out, const EigenVector<Eigen::Matrix<SReal, Eigen::Dynamic, 1>>& v)
{
    for (Index i = 0, s = v.size(); i < s; ++i)
    {
        if (i) out << ' ';
        out << v.vector()[i];
    }
    return out;
}

template class SOFA_LINEARALGEBRA_API EigenVector<Eigen::Matrix<SReal, Eigen::Dynamic, 1 > >;

} //namespace sofa::linearalgebra::eigen
