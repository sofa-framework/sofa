/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_INL
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_INL

#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

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
    nCol = 0;
    nBlocRow = M.rowBSize()*3;
    nBlocCol = 0;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*3);
    colsValue.reserve(M.colsValue.size()*3);

    int vid = 0;
    for (unsigned int rowId = 0; rowId < M.rowIndex.size(); ++rowId)
    {
        int i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (int lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                int j = M.colsIndex[xj] * 3;
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
    nCol = 0;
    nBlocRow = M.rowBSize()*3;
    nBlocCol = 0;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*3);
    colsValue.reserve(M.colsValue.size()*3);

    int vid = 0;
    for (unsigned int rowId = 0; rowId < M.rowIndex.size(); ++rowId)
    {
        int i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (int lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                int j = M.colsIndex[xj] * 3;
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
    nCol = 0;
    nBlocRow = M.rowBSize()*3;
    nBlocCol = 0;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*3);
    colsValue.reserve(M.colsValue.size()*3);

    int vid = 0;
    for (unsigned int rowId = 0; rowId < M.rowIndex.size(); ++rowId)
    {
        int i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (int lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                int j = M.colsIndex[xj] * 3;
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
    nCol = 0;
    nBlocRow = M.rowBSize()*3;
    nBlocCol = 0;
    rowIndex.clear();
    rowBegin.clear();
    colsIndex.clear();
    colsValue.clear();
    compressed = true;
    btemp.clear();
    rowIndex.reserve(M.rowIndex.size()*3);
    rowBegin.reserve(M.rowBegin.size()*3);
    colsIndex.reserve(M.colsIndex.size()*3);
    colsValue.reserve(M.colsValue.size()*3);

    int vid = 0;
    for (unsigned int rowId = 0; rowId < M.rowIndex.size(); ++rowId)
    {
        int i = M.rowIndex[rowId] * 3;

        Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);

        for (int lb = 0; lb<3 ; lb++)
        {
            rowIndex.push_back(i+lb);
            rowBegin.push_back(vid);

            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                int j = M.colsIndex[xj] * 3;
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
