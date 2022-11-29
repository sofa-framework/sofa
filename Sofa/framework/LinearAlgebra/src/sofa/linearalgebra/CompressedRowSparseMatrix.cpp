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
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_CPP
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/helper/narrow_cast.h>

namespace sofa::linearalgebra
{

template<class TMatrix, sofa::Size L, sofa::Size C, class real>
void filterValuesFromBlocs(TMatrix& self, CompressedRowSparseMatrix<type::Mat<L, C, real> >& M, typename TMatrix::filter_fn* filter, const typename TMatrix::Block& ref)
{
    M.compress();
    self.nBlocRow = 1;
    self.nBlocCol = 1;
    self.rowIndex.clear();
    self.rowBegin.clear();
    self.colsIndex.clear();
    self.colsValue.clear();
    self.btemp.clear();
    self.rowIndex.reserve(M.rowIndex.size() * 3);
    self.rowBegin.reserve(M.rowBegin.size() * 3);
    self.colsIndex.reserve(M.colsIndex.size() * 9);
    self.colsValue.reserve(M.colsValue.size() * 9);

    Index vid = 0;
    for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
    {
        Index i = M.rowIndex[rowId] * L;

        typename TMatrix::Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId + 1]);

        for (sofa::Size lb = 0; lb < C; lb++)
        {
            self.rowIndex.push_back(i + lb);
            self.rowBegin.push_back(vid);

            for (Index xj = rowRange.begin(); xj < (decltype(xj))rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj] * C;
                const type::Mat<L, C, real>& b = M.colsValue[xj];
                const auto& rowB = b[lb];

                for (sofa::Size c = 0; c < C; ++c)
                {
                    auto val = sofa::helper::narrow_cast<typename TMatrix::Block>(rowB[c]);
                    if ((*filter)(i + lb, j + c, val, ref))
                    {
                        self.colsIndex.push_back(j + c);
                        self.colsValue.push_back(sofa::helper::narrow_cast<typename TMatrix::Block>(b[lb][c]));
                        ++vid;
                    }
                }
            }

            if ((decltype(vid))self.rowBegin.back() == vid)   // row was empty
            {
                self.rowIndex.pop_back();
                self.rowBegin.pop_back();
            }
        }
    }
    self.rowBegin.push_back(vid); // end of last row
}

template <> template <>
void CompressedRowSparseMatrix<double>::filterValues(CompressedRowSparseMatrix<type::Mat<3, 3, double> >& M, filter_fn* filter, const Block& ref)
{
    filterValuesFromBlocs(*this, M, filter, ref);
}

template <> template <>
void CompressedRowSparseMatrix<double>::filterValues(CompressedRowSparseMatrix<type::Mat<3, 3, float> >& M, filter_fn* filter, const Block& ref)
{
    filterValuesFromBlocs(*this, M, filter, ref);
}

template <> template <>
void CompressedRowSparseMatrix<float>::filterValues(CompressedRowSparseMatrix<type::Mat<3, 3, float> >& M, filter_fn* filter, const Block& ref)
{
    filterValuesFromBlocs(*this, M, filter, ref);
}

template <> template <>
void CompressedRowSparseMatrix<float>::filterValues(CompressedRowSparseMatrix<type::Mat<3, 3, double> >& M, filter_fn* filter, const Block& ref)
{
    filterValuesFromBlocs(*this, M, filter, ref);
}


template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<float>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<double>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<1, 1, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<1, 1, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<2, 2, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<2, 2, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<3, 3, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<3, 3, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<4, 4, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<4, 4, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<6, 6, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<6, 6, double> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<8, 8, float> >;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrix<type::Mat<8, 8, double> >;

} // namespace sofa::linearalgebra
