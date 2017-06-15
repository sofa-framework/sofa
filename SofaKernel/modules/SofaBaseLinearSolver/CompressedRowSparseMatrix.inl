/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_INL
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_INL

#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template <> template <>
inline void CompressedRowSparseMatrix<double>::filterValues(CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >& M, filter_fn* filter, const Bloc& ref)
{
    M.compress();
    nRow = M.rowSize();
    nCol = M.colSize();
    nBlocRow = 1;
    nBlocCol = 1;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*9);
    colsValue.reserve(M.colsValue.size()*9);

    Index vid = 0;
    for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
    {
        Index i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (Index lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj] * 3;
                defaulttype::Mat<3,3,double> b = M.colsValue[xj];
                if ((*filter)(i+lb,j+0,b[lb][0],ref))
                {
                    colsIndex.push_back(j+0);
                    colsValue.push_back(b[lb][0]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+1,b[lb][1],ref))
                {
                    colsIndex.push_back(j+1);
                    colsValue.push_back(b[lb][1]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+2,b[lb][2],ref))
                {
                    colsIndex.push_back(j+2);
                    colsValue.push_back(b[lb][2]);
                    ++vid;
                }
            }

            if (rowBegin.back() == vid)   // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
    }
    rowBegin.push_back(vid); // end of last row
}

template <> template <>
inline void CompressedRowSparseMatrix<double>::filterValues(CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >& M, filter_fn* filter, const Bloc& ref)
{
    M.compress();
    nRow = M.rowSize();
    nCol = M.colSize();
    nBlocRow = 1;
    nBlocCol = 1;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*9);
    colsValue.reserve(M.colsValue.size()*9);

    Index vid = 0;
    for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
    {
        Index i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (Index lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj] * 3;
                defaulttype::Mat<3,3,double> b = M.colsValue[xj];
                if ((*filter)(i+lb,j+0,b[lb][0],ref))
                {
                    colsIndex.push_back(j+0);
                    colsValue.push_back(b[lb][0]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+1,b[lb][1],ref))
                {
                    colsIndex.push_back(j+1);
                    colsValue.push_back(b[lb][1]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+2,b[lb][2],ref))
                {
                    colsIndex.push_back(j+2);
                    colsValue.push_back(b[lb][2]);
                    ++vid;
                }
            }

            if (rowBegin.back() == vid)   // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
    }
    rowBegin.push_back(vid); // end of last row
}

template <> template <>
inline void CompressedRowSparseMatrix<float>::filterValues(CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >& M, filter_fn* filter, const Bloc& ref)
{
    M.compress();
    nRow = M.rowSize();
    nCol = M.colSize();
    nBlocRow = 1;
    nBlocCol = 1;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*9);
    colsValue.reserve(M.colsValue.size()*9);

    Index vid = 0;
    for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
    {
        Index i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (Index lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj] * 3;
                defaulttype::Mat<3,3,float> b = M.colsValue[xj];
                if ((*filter)(i+lb,j+0,b[lb][0],ref))
                {
                    colsIndex.push_back(j+0);
                    colsValue.push_back(b[lb][0]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+1,b[lb][1],ref))
                {
                    colsIndex.push_back(j+1);
                    colsValue.push_back(b[lb][1]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+2,b[lb][2],ref))
                {
                    colsIndex.push_back(j+2);
                    colsValue.push_back(b[lb][2]);
                    ++vid;
                }
            }

            if (rowBegin.back() == vid)   // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
    }
    rowBegin.push_back(vid); // end of last row
}

template <> template <>
inline void CompressedRowSparseMatrix<float>::filterValues(CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >& M, filter_fn* filter, const Bloc& ref)
{
    M.compress();
    nRow = M.rowSize();
    nCol = M.colSize();
    nBlocRow = 1;
    nBlocCol = 1;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*9);
    colsValue.reserve(M.colsValue.size()*9);

    Index vid = 0;
    for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
    {
        Index i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (Index lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj] * 3;
                defaulttype::Mat<3,3,float> b = M.colsValue[xj];
                if ((*filter)(i+lb,j+0,b[lb][0],ref))
                {
                    colsIndex.push_back(j+0);
                    colsValue.push_back(b[lb][0]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+1,b[lb][1],ref))
                {
                    colsIndex.push_back(j+1);
                    colsValue.push_back(b[lb][1]);
                    ++vid;
                }
                if ((*filter)(i+lb,j+2,b[lb][2],ref))
                {
                    colsIndex.push_back(j+2);
                    colsValue.push_back(b[lb][2]);
                    ++vid;
                }
            }

            if (rowBegin.back() == vid)   // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
    }
    rowBegin.push_back(vid); // end of last row
}


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
