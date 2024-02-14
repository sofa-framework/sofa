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
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIXMECHANICAL_CPP
#include <sofa/linearalgebra/CompressedRowSparseMatrixMechanical.h>

namespace sofa::linearalgebra
{


template <class TMatrix, class TBlockMatrix>
void addBlockMat(TMatrix& self, Index row, Index col, const TBlockMatrix& _M)
{
    if (row % TBlockMatrix::nbLines == 0 && col % TBlockMatrix::nbCols == 0)
    {
        *self.wblock(row / TBlockMatrix::nbLines, col / TBlockMatrix::nbCols, true) += _M;
    }
    else
    {
        self.linearalgebra::BaseMatrix::add(row, col, _M);
    }
}

template <>
void CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >::add(Index row, Index col, const type::Mat3x3d& _M)
{
    addBlockMat(*this, row, col, _M);
}

template <>
void CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >::add(Index row, Index col, const type::Mat3x3f& _M)
{
    addBlockMat(*this, row, col, _M);
}

template <>
void CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >::add(Index row, Index col, const type::Mat3x3d& _M)
{
    addBlockMat(*this, row, col, _M);
}

template <>
void CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >::add(Index row, Index col, const type::Mat3x3f& _M)
{
    addBlockMat(*this, row, col, _M);
}

using namespace sofa::type;

template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<float>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat1x1f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat2x2f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat3x3f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat4x4f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat<6, 6, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat<8, 8, float> >;

template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<double>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat1x1d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat2x2d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat3x3d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat4x4d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat<6, 6, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<Mat<8, 8, double> >;

} // namespace sofa::linearalgebra
