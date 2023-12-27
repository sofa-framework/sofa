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


template<class RealDest, class RealSrc>
void filterValuesImpl(
    CompressedRowSparseMatrixMechanical<RealDest>& dest,
    CompressedRowSparseMatrixMechanical<type::Mat<3, 3, RealSrc> >& src,
    typename CompressedRowSparseMatrixMechanical<RealDest>::filter_fn* filter,
    const RealDest ref, const bool keepEmptyRows)
{
    src.compress();
    dest.nRow = src.rowSize();
    dest.nCol = src.colSize();
    dest.nBlockRow = src.rowSize();
    dest.nBlockCol = src.colSize();
    dest.rowIndex.clear();
    dest.rowBegin.clear();
    dest.colsIndex.clear();
    dest.colsValue.clear();
    dest.btemp.clear();
    dest.skipCompressZero = true;
    dest.rowIndex.reserve(src.rowIndex.size() * 3);
    dest.rowBegin.reserve(src.rowBegin.size() * 3);
    dest.colsIndex.reserve(src.colsIndex.size() * 9);
    dest.colsValue.reserve(src.colsValue.size() * 9);

    Index vid = 0;
    for (std::size_t rowId = 0; rowId < src.rowIndex.size(); ++rowId)
    {
        const Index i = src.rowIndex[rowId] * 3;

        typename CompressedRowSparseMatrixMechanical<RealDest>::Range rowRange(src.rowBegin[rowId], src.rowBegin[rowId + 1]);

        for (Index lb = 0; lb < 3; lb++)
        {
            dest.rowIndex.push_back(i + lb);
            dest.rowBegin.push_back(vid);

            for (std::size_t xj = static_cast<std::size_t>(rowRange.begin()); xj < static_cast<std::size_t>(rowRange.end()); ++xj)
            {
                const Index j = src.colsIndex[xj] * 3;
                type::Mat<3, 3, RealDest> b = src.colsValue[xj];
                if ((*filter)(i + lb, j + 0, b[lb][0], ref))
                {
                    dest.colsIndex.push_back(j + 0);
                    dest.colsValue.push_back(b[lb][0]);
                    ++vid;
                }
                if ((*filter)(i + lb, j + 1, b[lb][1], ref))
                {
                    dest.colsIndex.push_back(j + 1);
                    dest.colsValue.push_back(b[lb][1]);
                    ++vid;
                }
                if ((*filter)(i + lb, j + 2, b[lb][2], ref))
                {
                    dest.colsIndex.push_back(j + 2);
                    dest.colsValue.push_back(b[lb][2]);
                    ++vid;
                }
            }

            if (!keepEmptyRows && dest.rowBegin.back() == vid)   // row was empty
            {
                dest.rowIndex.pop_back();
                dest.rowBegin.pop_back();
            }
        }
    }
    dest.rowBegin.push_back(vid); // end of last row
}

template <> template <>
void CompressedRowSparseMatrixMechanical<double>::filterValues(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows)
{
    filterValuesImpl(*this, M, filter, ref, keepEmptyRows);
}

template <> template <>
void CompressedRowSparseMatrixMechanical<double>::filterValues(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows)
{
    filterValuesImpl(*this, M, filter, ref, keepEmptyRows);
}

template <> template <>
void CompressedRowSparseMatrixMechanical<float>::filterValues(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows)
{
    filterValuesImpl(*this, M, filter, ref, keepEmptyRows);
}

template <> template <>
void CompressedRowSparseMatrixMechanical<float>::filterValues(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows)
{
    filterValuesImpl(*this, M, filter, ref, keepEmptyRows);
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
