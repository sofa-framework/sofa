/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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

template <>
template<typename RB, typename RVB, typename RVI, typename MB, typename MVB, typename MVI >
void CompressedRowSparseMatrix<double>::mulTranspose( CompressedRowSparseMatrix<RB,RVB,RVI>& res, const CompressedRowSparseMatrix<MB,MVB,MVI>& m ) const
{
    assert( rowSize() == m.rowSize() );

    // must already be compressed, since matrices are const they cannot be modified
    //compress();
    //m.compress();
    ((Matrix*)this)->compress();  /// \warning this violates the const-ness of the method
    ((CompressedRowSparseMatrix<MB,MVB,MVI>*)&m)->compress();  /// \warning this violates the const-ness of the parameter


    res.resize( this->nCol, m.nCol );  // clear and resize the result

    if( m.rowIndex.empty() ) return; // if m is null

    for( Index xi = 0 ; xi < (Index)rowIndex.size() ; ++xi )  // for each non-null transpose block column
    {
        unsigned mr = 0; // block row index in m

        Index col = rowIndex[xi];      // block col (transposed col = row)

        Range rowRange( rowBegin[xi], rowBegin[xi+1] );
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
        {
            Index row = colsIndex[xj];     // block row (transposed row = col)
            const Bloc& b = colsValue[xj]; // block value

            // find the non-null row in m, if any
            while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
            if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

            // Accumulate  res[row] += b^T * m[col]
            Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
            for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
            {
                Index mcol = m.colsIndex[mj];     // column index of the non-null block
                *res.wbloc(row,mcol,true) += (b * m.colsValue[mj]);  // find the matching bloc in res, and accumulate the block product
            }
        }
    }
    res.compress();
}

template <>
template<typename RB, typename RVB, typename RVI, typename MB, typename MVB, typename MVI >
void CompressedRowSparseMatrix<float>::mulTranspose( CompressedRowSparseMatrix<RB,RVB,RVI>& res, const CompressedRowSparseMatrix<MB,MVB,MVI>& m ) const
{

    assert( rowSize() == m.rowSize() );

    // must already be compressed, since matrices are const they cannot be modified
    //compress();
    //m.compress();
    ((Matrix*)this)->compress();  /// \warning this violates the const-ness of the method
    ((CompressedRowSparseMatrix<MB,MVB,MVI>*)&m)->compress();  /// \warning this violates the const-ness of the parameter


    res.resize( this->nCol, m.nCol );  // clear and resize the result

    if( m.rowIndex.empty() ) return; // if m is null

    for( Index xi = 0 ; xi < (Index)rowIndex.size() ; ++xi )  // for each non-null transpose block column
    {
        unsigned mr = 0; // block row index in m

        Index col = rowIndex[xi];      // block col (transposed col = row)

        Range rowRange( rowBegin[xi], rowBegin[xi+1] );
        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
        {
            Index row = colsIndex[xj];     // block row (transposed row = col)
            const Bloc& b = colsValue[xj]; // block value

            // find the non-null row in m, if any
            while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
            if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

            // Accumulate  res[row] += b^T * m[col]
            Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
            for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
            {
                Index mcol = m.colsIndex[mj];     // column index of the non-null block
                *res.wbloc(row,mcol,true) += (b * m.colsValue[mj]);  // find the matching bloc in res, and accumulate the block product
            }
        }
    }
    res.compress();
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
