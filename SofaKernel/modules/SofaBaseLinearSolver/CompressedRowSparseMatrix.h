/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_H
#include "config.h"

#include <sofa/defaulttype/BaseMatrix.h>
#include <SofaBaseLinearSolver/MatrixExpr.h>
#include <SofaBaseLinearSolver/matrix_bloc_traits.h>
#include "FullVector.h"
#include <algorithm>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define SPARSEMATRIX_CHECK
//#define SPARSEMATRIX_VERBOSE

/// This pattern is used to force compilation of code fragment that depend on the definition of
/// the "define". In the following, use if(EMIT_EXTRA_MESSAGE) instead of #ifdef
#ifdef SPARSEMATRIX_VERBOSE
#define EMIT_EXTRA_MESSAGE true
#else
#define EMIT_EXTRA_MESSAGE false
#endif

template<typename TBloc, typename TVecBloc = helper::vector<TBloc>, typename TVecIndex = helper::vector<int> >
class CompressedRowSparseMatrix : public defaulttype::BaseMatrix
{
public:
    typedef CompressedRowSparseMatrix<TBloc,TVecBloc,TVecIndex> Matrix;
    typedef TBloc Bloc;
    typedef matrix_bloc_traits<Bloc> traits;
    typedef typename traits::Real Real;
    enum { NL = traits::NL };  ///< Number of rows of a block
    enum { NC = traits::NC };  ///< Number of columns of a block

    typedef Matrix Expr;
    typedef CompressedRowSparseMatrix<double> matrix_type;
    enum { category = MATRIX_SPARSE };
    enum { operand = 1 };

    typedef TVecBloc VecBloc;
    typedef TVecIndex VecIndex;
    struct IndexedBloc
    {
        Index l,c;
        Bloc value;
        IndexedBloc() {}
        IndexedBloc(Index i, Index j) : l(i), c(j) {}
        IndexedBloc(Index i, Index j, const Bloc& v) : l(i), c(j), value(v) {}
        bool operator < (const IndexedBloc& b) const
        {
            return (l < b.l) || (l == b.l && c < b.c);
        }
        bool operator <= (const IndexedBloc& b) const
        {
            return (l < b.l) || (l == b.l && c <= b.c);
        }
        bool operator > (const IndexedBloc& b) const
        {
            return (l > b.l) || (l == b.l && c > b.c);
        }
        bool operator >= (const IndexedBloc& b) const
        {
            return (l > b.l) || (l == b.l && c >= b.c);
        }
        bool operator == (const IndexedBloc& b) const
        {
            return (l == b.l) && (c == b.c);
        }
        bool operator != (const IndexedBloc& b) const
        {
            return (l != b.l) || (c != b.c);
        }
    };
    typedef helper::vector<IndexedBloc> VecIndexedBloc;

    static void split_row_index(Index& index, Index& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(Index& index, Index& modulo) { bloc_index_func<NC>::split(index, modulo); }

    class Range : public std::pair<Index, Index>
    {
        typedef std::pair<Index, Index> Inherit;
    public:
        Range() : Inherit(0,0) {}
        Range(Index begin, Index end) : Inherit(begin,end) {}
        Index begin() const { return this->first; }
        Index end() const { return this->second; }
        void setBegin(Index i) { this->first = i; }
        void setEnd(Index i) { this->second = i; }
        bool empty() const { return begin() == end(); }
        Index size() const { return end()-begin(); }
        typename VecBloc::iterator begin(VecBloc& b) const { return b.begin() + begin(); }
        typename VecBloc::iterator end  (VecBloc& b) const { return b.end  () + end  (); }
        typename VecBloc::const_iterator begin(const VecBloc& b) const { return b.begin() + begin(); }
        typename VecBloc::const_iterator end  (const VecBloc& b) const { return b.end  () + end  (); }
        typename VecIndex::iterator begin(VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::iterator end  (VecIndex& b) const { return b.end  () + end  (); }
        typename VecIndex::const_iterator begin(const VecIndex& b) const { return b.begin() + begin(); }
        typename VecIndex::const_iterator end  (const VecIndex& b) const { return b.end  () + end  (); }
        void operator++() { ++first; }
        void operator++(int) { ++first; }
    };

    static bool sortedFind(const VecIndex& v, Range in, Index val, Index& result)
    {
        if (in.empty()) return false;
        Index candidate = (result >= in.begin() && result < in.end()) ? result : ((in.begin() + in.end()) >> 1);
        for(;;)
        {
            Index i = v[candidate];
            if (i == val) { result = candidate; return true; }
            if (i < val)  in.setBegin(candidate+1);
            else          in.setEnd(candidate);
            if (in.empty()) break;
            candidate = (in.begin() + in.end()) >> 1;
        }
        return false;
    }

    static bool sortedFind(const VecIndex& v, Index val, Index& result)
    {
        return sortedFind(v, Range(0,(Index)v.size()), val, result);
    }

public :
    // size
    Index nRow,nCol;         ///< Mathematical size of the matrix, in scalars
    Index nBlocRow,nBlocCol; ///< Mathematical size of the matrix, in blocks.

    // compressed sparse data structure
    VecIndex rowIndex;   ///< indices of non-empty block rows
    VecIndex rowBegin;   ///< column indices of non-empty blocks in each row. The column indices of the non-empty block within the i-th non-empty row are all the colsIndex[j],  j  in [rowBegin[i],rowBegin[i+1])
    VecIndex colsIndex;  ///< column indices of all the non-empty blocks, sorted by increasing row index and column index
    VecBloc  colsValue;  ///< values of the non-empty blocks, in the same order as in colsIndex

    // additional storage to make block insertion more efficient
    VecIndexedBloc btemp; ///< unsorted blocks and their indices
    bool compressed;      ///< true if the additional storage is empty or has been transfered to the compressed data structure

    // Temporary vectors used during compression
    VecIndex oldRowIndex;
    VecIndex oldRowBegin;
    VecIndex oldColsIndex;
    VecBloc  oldColsValue;
public:
    CompressedRowSparseMatrix()
        : nRow(0), nCol(0), nBlocRow(0), nBlocCol(0), compressed(true)
    {
    }

    CompressedRowSparseMatrix(Index nbRow, Index nbCol)
        : nRow(nbRow), nCol(nbCol),
          nBlocRow((nbRow + NL-1) / NL), nBlocCol((nbCol + NC-1) / NC),
          compressed(true)
    {
    }

    ~CompressedRowSparseMatrix()
    {
        this->clear();
    }

    /// \returns the number of row blocs
    Index rowBSize() const
    {
        return nBlocRow;
    }

    /// \returns the number of col blocs
    Index colBSize() const
    {
        return nBlocCol;
    }

    const VecIndex& getRowIndex() const { return rowIndex; }
    const VecIndex& getRowBegin() const { return rowBegin; }
    Range getRowRange(Index id) const { return Range(rowBegin[id], rowBegin[id+1]); }
    const VecIndex& getColsIndex() const { return colsIndex; }
    const VecBloc& getColsValue() const { return colsValue; }

    void resizeBloc(Index nbBRow, Index nbBCol)
    {
        if (nBlocRow == nbBRow && nBlocRow == nbBCol)
        {
            // just clear the matrix
            for (Index i=0; i < (Index)colsValue.size(); ++i)
                traits::clear(colsValue[i]);
            compressed = colsValue.empty();
            btemp.clear();
        }
        else
        {
            msg_info_when(EMIT_EXTRA_MESSAGE)
                    << ": resize("<<nbBRow<<"*"<<NL<<","<<nbBCol<<"*"<<NC<<")" ;

            nRow = nbBRow*NL;
            nCol = nbBCol*NC;
            nBlocRow = nbBRow;
            nBlocCol = nbBCol;
            rowIndex.clear();
            rowBegin.clear();
            colsIndex.clear();
            colsValue.clear();
            compressed = true;
            btemp.clear();
        }
    }

    virtual void compress()
    {
        if (compressed && btemp.empty()) return;
        if (!btemp.empty())
        {
            dmsg_info_when(EMIT_EXTRA_MESSAGE)
                    << "("<<rowSize()<<","<<colSize()<<"): sort "<<btemp.size()<<" temp blocs." ;
            std::sort(btemp.begin(),btemp.end());
            dmsg_info_when(EMIT_EXTRA_MESSAGE)
                    << "("<<rowSize()<<","<<colSize()<<"): blocs sorted." ;
        }
        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        oldColsIndex.swap(colsIndex);
        oldColsValue.swap(colsValue);
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        rowIndex.reserve(oldRowIndex.empty() ? nBlocRow : oldRowIndex.size());
        rowBegin.reserve((oldRowIndex.empty() ? nBlocRow : oldRowIndex.size())+1);
        colsIndex.reserve(oldColsIndex.size() + btemp.size());
        colsValue.reserve(oldColsIndex.size() + btemp.size());
        const Index oldNRow = (Index)oldRowIndex.size();
        const Index EndRow = nBlocRow;
        const Index EndCol = nBlocCol;
        //const Index EndVal = oldColsIndex.size();
        Index inRowId = 0;
        Index inRowIndex = (inRowId < oldNRow ) ? oldRowIndex[inRowId] : EndRow;
        typename VecIndexedBloc::const_iterator itbtemp = btemp.begin(), endbtemp = btemp.end();
        Index bRowIndex = (itbtemp != endbtemp) ? itbtemp->l : EndRow;
        Index outValId = 0;
        while (inRowIndex < EndRow || bRowIndex < EndRow)
        {
            dmsg_info_when(EMIT_EXTRA_MESSAGE)
                    << "("<<rowSize()<<","<<colSize()<<"): inRowIndex = "<<inRowIndex<<" , bRowIndex = "<<bRowIndex<<"" ;
            if (inRowIndex < bRowIndex)
            {
                // this row contains values only from old*
                rowIndex.push_back(inRowIndex);
                rowBegin.push_back(outValId);
                Range inRow( oldRowBegin[inRowId], oldRowBegin[inRowId+1] );
                while (!inRow.empty())
                {
                    if (!traits::empty(oldColsValue[inRow.begin()]))
                    {
                        colsIndex.push_back(oldColsIndex[inRow.begin()]);
                        colsValue.push_back(oldColsValue[inRow.begin()]);
                        ++outValId;
                    }
                    ++inRow;
                }
                ++inRowId;
                inRowIndex = (inRowId < oldNRow ) ? oldRowIndex[inRowId] : EndRow;
            }
            else if (inRowIndex > bRowIndex)
            {
                // this row contains values only from btemp
                rowIndex.push_back(bRowIndex);
                rowBegin.push_back(outValId);
                while (itbtemp != endbtemp && itbtemp->l == bRowIndex)
                {
                    Index bColIndex = itbtemp->c;
                    colsIndex.push_back(bColIndex);
                    colsValue.push_back(itbtemp->value);
                    ++itbtemp;
                    Bloc& value = colsValue.back();
                    while (itbtemp != endbtemp && itbtemp->c == bColIndex && itbtemp->l == bRowIndex)
                    {
                        value += itbtemp->value;
                        ++itbtemp;
                    }
                    ++outValId;
                }
                bRowIndex = (itbtemp != endbtemp) ? itbtemp->l : EndRow;
            }
            else
            {
                // this row mixes values from old* and btemp
                rowIndex.push_back(inRowIndex);
                rowBegin.push_back(outValId);
                Range inRow( oldRowBegin[inRowId], oldRowBegin[inRowId+1] );
                Index inColIndex = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : EndCol;
                Index bColIndex = (itbtemp != endbtemp && itbtemp->l == inRowIndex) ? itbtemp->c : EndCol;
                while (inColIndex < EndCol || bColIndex < EndCol)
                {
                    if (inColIndex < bColIndex)
                    {
                        if (!traits::empty(oldColsValue[inRow.begin()]))
                        {
                            colsIndex.push_back(inColIndex);
                            colsValue.push_back(oldColsValue[inRow.begin()]);
                            ++outValId;
                        }
                        ++inRow;
                        inColIndex = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : EndCol;
                    }
                    else if (inColIndex > bColIndex)
                    {
                        colsIndex.push_back(bColIndex);
                        colsValue.push_back(itbtemp->value);
                        ++itbtemp;
                        Bloc& value = colsValue.back();
                        while (itbtemp != endbtemp && itbtemp->c == bColIndex && itbtemp->l == bRowIndex)
                        {
                            value += itbtemp->value;
                            ++itbtemp;
                        }
                        bColIndex = (itbtemp != endbtemp && itbtemp->l == bRowIndex) ? itbtemp->c : EndCol;
                        ++outValId;
                    }
                    else
                    {
                        colsIndex.push_back(inColIndex);
                        colsValue.push_back(oldColsValue[inRow.begin()]);
                        ++inRow;
                        inColIndex = (!inRow.empty()) ? oldColsIndex[inRow.begin()] : EndCol;
                        Bloc& value = colsValue.back();
                        while (itbtemp != endbtemp && itbtemp->c == bColIndex && itbtemp->l == bRowIndex)
                        {
                            value += itbtemp->value;
                            ++itbtemp;
                        }
                        bColIndex = (itbtemp != endbtemp && itbtemp->l == bRowIndex) ? itbtemp->c : EndCol;
                        ++outValId;
                    }
                }
                ++inRowId;
                inRowIndex = (inRowId < oldNRow ) ? oldRowIndex[inRowId] : EndRow;
                bRowIndex = (itbtemp != endbtemp) ? itbtemp->l : EndRow;
            }
        }
        rowBegin.push_back(outValId);
        btemp.clear();
        compressed = true;
    }

    void swap(Matrix& m)
    {
        Index t;
        t = nRow; nRow = m.nRow; m.nRow = t;
        t = nCol; nCol = m.nCol; m.nCol = t;
        t = nBlocRow; nBlocRow = m.nBlocRow; m.nBlocRow = t;
        t = nBlocCol; nBlocCol = m.nBlocCol; m.nBlocCol = t;
        bool b;
        b = compressed; compressed = m.compressed; m.compressed = b;
        rowIndex.swap(m.rowIndex);
        rowBegin.swap(m.rowBegin);
        colsIndex.swap(m.colsIndex);
        rowBegin.swap(m.rowBegin);
        colsIndex.swap(m.colsIndex);
        colsValue.swap(m.colsValue);
        btemp.swap(m.btemp);
    }

    /// Make sure all rows have an entry even if they are empty
    void fullRows()
    {
        compress();
        if (rowIndex.size() >= nRow) return;
        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        rowIndex.resize(nRow);
        rowBegin.resize(nRow+1);
        for (Index i=0; i<nRow; ++i) rowIndex[i] = i;
        Index j = 0;
        Index b = 0;
        for (Index i=0; i<oldRowIndex.size(); ++i)
        {
            b = oldRowBegin[i];
            for (; j<=oldRowIndex[i]; ++j)
                rowBegin[j] = b;
        }
        b = oldRowBegin[oldRowBegin.size()-1];
        for (; j<=nRow; ++j)
            rowBegin[j] = b;
    }

    /// Make sure all diagonal entries are present even if they are zero
    void fullDiagonal()
    {
        compress();
        Index ndiag = 0;
        for (Index r=0; r<rowIndex.size(); ++r)
        {
            Index i = rowIndex[r];
            Index b = rowBegin[r];
            Index e = rowBegin[r+1];
            Index t = b;
            while (b < e && colsIndex[t] != i)
            {
                if (colsIndex[t] < i)
                    b = t+1;
                else
                    e = t;
                t = (b+e)>>1;
            }
            if (b<e) ++ndiag;
        }
        if (ndiag == nRow) return;

        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        oldColsIndex.swap(colsIndex);
        oldColsValue.swap(colsValue);
        rowIndex.resize(nRow);
        rowBegin.resize(nRow+1);
        colsIndex.resize(oldColsIndex.size()+nRow-ndiag);
        colsValue.resize(oldColsValue.size()+nRow-ndiag);
        Index nv = 0;
        for (Index i=0; i<nRow; ++i) rowIndex[i] = i;
        Index j = 0;
        for (Index i=0; i<oldRowIndex.size(); ++i)
        {
            for (; j<oldRowIndex[i]; ++j)
            {
                rowBegin[j] = nv;
                colsIndex[nv] = j;
                traits::clear(colsValue[nv]);
                ++nv;
            }
            rowBegin[j] = nv;
            Index b = oldRowBegin[i];
            Index e = oldRowBegin[i+1];
            for (; b<e && oldColsIndex[b] < j; ++b)
            {
                colsIndex[nv] = oldColsIndex[b];
                colsValue[nv] = oldColsValue[b];
                ++nv;
            }
            if (b>=e || oldColsIndex[b] > j)
            {
                colsIndex[nv] = j;
                traits::clear(colsValue[nv]);
                ++nv;
            }
            for (; b<e; ++b)
            {
                colsIndex[nv] = oldColsIndex[b];
                colsValue[nv] = oldColsValue[b];
                ++nv;
            }
            ++j;
        }
        for (; j<nRow; ++j)
        {
            rowBegin[j] = nv;
            colsIndex[nv] = j;
            traits::clear(colsValue[nv]);
            ++nv;
        }
        rowBegin[j] = nv;
    }

    /// Add the given base to all indices.
    /// Use 1 to convert do Fortran 1-based notation.
    /// Note that the matrix will no longer be valid
    /// from the point of view of C/C++ codes. You need
    /// to call again with -1 as base to undo it.
    void shiftIndices(Index base)
    {
        for (Index i=0; i<rowIndex.size(); ++i)
            rowIndex[i] += base;
        for (Index i=0; i<rowBegin.size(); ++i)
            rowBegin[i] += base;
        for (Index i=0; i<colsIndex.size(); ++i)
            colsIndex[i] += base;
    }

    // filtering-out part of a matrix
    typedef bool filter_fn    (Index   i  , Index   j  , Bloc& val, const Bloc&   ref  );
    static bool       nonzeros(Index /*i*/, Index /*j*/, Bloc& val, const Bloc& /*ref*/) { return (!traits::empty(val)); }
    static bool       nonsmall(Index /*i*/, Index /*j*/, Bloc& val, const Bloc&   ref  )
    {
        for (Index bi = 0; bi < NL; ++bi)
            for (Index bj = 0; bj < NC; ++bj)
                if (helper::rabs(traits::v(val, bi, bj)) >= ref) return true;
        return false;
    }
    static bool upper         (Index   i  , Index   j  , Bloc& val, const Bloc& /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (Index bi = 1; bi < NL; ++bi)
                for (Index bj = 0; bj < bi; ++bj)
                    traits::v(val, bi, bj) = 0;
        }
        return i*NL <= j*NC;
    }
    static bool lower         (Index   i  , Index   j  , Bloc& val, const Bloc& /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (Index bi = 0; bi < NL-1; ++bi)
                for (Index bj = bi+1; bj < NC; ++bj)
                    traits::v(val, bi, bj) = 0;
        }
        return i*NL >= j*NC;
    }
    static bool upper_nonzeros(Index   i  , Index   j  , Bloc& val, const Bloc&   ref  ) { return upper(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool lower_nonzeros(Index   i  , Index   j  , Bloc& val, const Bloc&   ref  ) { return lower(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool upper_nonsmall(Index   i  , Index   j  , Bloc& val, const Bloc&   ref  ) { return upper(i,j,val,ref) && nonsmall(i,j,val,ref); }
    static bool lower_nonsmall(Index   i  , Index   j  , Bloc& val, const Bloc&   ref  ) { return lower(i,j,val,ref) && nonsmall(i,j,val,ref); }

    template<class TMatrix>
    void filterValues(TMatrix& M, filter_fn* filter = &nonzeros, const Bloc& ref = Bloc())
    {
        M.compress();
        nRow = M.rowSize();
        nCol = M.colSize();
        nBlocRow = M.rowBSize();
        nBlocCol = M.colBSize();
        rowIndex.clear();
        rowBegin.clear();
        colsIndex.clear();
        colsValue.clear();
        compressed = true;
        btemp.clear();
        rowIndex.reserve(M.rowIndex.size());
        rowBegin.reserve(M.rowBegin.size());
        colsIndex.reserve(M.colsIndex.size());
        colsValue.reserve(M.colsValue.size());

        Index vid = 0;
        for (Index rowId = 0; rowId < (Index)M.rowIndex.size(); ++rowId)
        {
            Index i = M.rowIndex[rowId];
            rowIndex.push_back(i);
            rowBegin.push_back(vid);
            Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj];
                Bloc b = M.colsValue[xj];
                if ((*filter)(i,j,b,ref))
                {
                    colsIndex.push_back(j);
                    colsValue.push_back(b);
                    ++vid;
                }
            }
            if (rowBegin.back() == vid) // row was empty
            {
                rowIndex.pop_back();
                rowBegin.pop_back();
            }
        }
        rowBegin.push_back(vid); // end of last row
    }

    template <class TMatrix>
    void copyNonZeros(TMatrix& M)
    {
        filterValues(M, nonzeros, Bloc());
    }

    template <class TMatrix>
    void copyNonSmall(TMatrix& M, const Bloc& ref)
    {
        filterValues(M, nonsmall, ref);
    }

    void copyUpper(Matrix& M)
    {
        filterValues(M, upper);
    }

    void copyLower(Matrix& M)
    {
        filterValues(M, lower);
    }

    template <class TMatrix>
    void copyUpperNonZeros(TMatrix& M)
    {
        filterValues(M, upper_nonzeros);
    }

    template <class TMatrix>
    void copyLowerNonZeros(TMatrix& M)
    {
        filterValues(M, lower_nonzeros);
    }

    void copyUpperNonSmall(Matrix& M, const Bloc& ref)
    {
        filterValues(M, upper_nonsmall, ref);
    }

    void copyLowerNonSmall(Matrix& M, const Bloc& ref)
    {
        filterValues(M, lower_nonsmall, ref);
    }

    const Bloc& bloc(Index i, Index j) const
    {
        static Bloc empty;
        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return colsValue[colId];
            }
        }
        return empty;
    }

    Bloc* wbloc(Index i, Index j, bool create = false)
    {
        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {

                dmsg_info_when(EMIT_EXTRA_MESSAGE)
                        << "("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<") found at "<<colId<<" (line "<<rowId<<")." ;

                return &colsValue[colId];
            }
        }
        if (create)
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
                dmsg_info_when(EMIT_EXTRA_MESSAGE)
                        << "("<<rowSize()<<","<<colSize()<<"): new temp bloc ("<<i<<","<<j<<")" ;

                btemp.push_back(IndexedBloc(i,j));
                traits::clear(btemp.back().value);
            }
            return &btemp.back().value;
        }
        return NULL;
    }

    ///< Mathematical size of the matrix
    Index rowSize() const
    {
        return nRow;
    }

    ///< Mathematical size of the matrix
    Index colSize() const
    {
        return nCol;
    }

    void resize(Index nbRow, Index nbCol)
    {
        if (EMIT_EXTRA_MESSAGE)
        {
            if (nbRow != rowSize() || nbCol != colSize())
                msg_info() << ": resize("<<nbRow<<","<<nbCol<<")" ;
        }

        resizeBloc((nbRow + NL-1) / NL, (nbCol + NC-1) / NC);
        nRow = nbRow;
        nCol = nbCol;
    }

    SReal element(Index i, Index j) const
    {
#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        ((Matrix*)this)->compress();  /// \warning this violates the const-ness of the method !
        return (SReal)traits::v(bloc(i, j), bi, bj);
    }

    void set(Index i, Index j, double v)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v ;

#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);

        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<")["<<bi<<","<<bj<<"] = "<<v ;

        traits::v(*wbloc(i,j,true), bi, bj) = (Real)v;
    }

    void add(Index i, Index j, double v)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v ;

#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);

        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<")["<<bi<<","<<bj<<"] += "<<v ;

        traits::v(*wbloc(i,j,true), bi, bj) += (Real)v;
    }

    void clear(Index i, Index j)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0" ;

#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize() || j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        compress();
        Bloc* b = wbloc(i,j,false);
        if (b)
            traits::v(*b, bi, bj) = 0;
    }

    void clearRow(Index i)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0" ;

#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize())
        {
            std::cerr << "ERROR: invalid write access to row "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        Index bi=0; split_row_index(i, bi);
        compress();
        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Bloc& b = colsValue[xj];
                for (Index bj = 0; bj < NC; ++bj)
                    traits::v(b, bi, bj) = 0;
            }
        }
    }

    void clearCol(Index j)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): col("<<j<<") = 0" ;

#ifdef SPARSEMATRIX_CHECK
        if (j >= colSize())
        {
            std::cerr << "ERROR: invalid write access to column "<<j<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        Index bj=0; split_col_index(j, bj);
        compress();
        for (Index i=0; i<nBlocRow; ++i)
        {
            Bloc* b = wbloc(i,j,false);
            if (b)
            {
                for (Index bi = 0; bi < NL; ++bi)
                    traits::v(*b, bi, bj) = 0;
            }
        }
    }

    void clearRowCol(Index i)
    {
        dmsg_info_when(EMIT_EXTRA_MESSAGE)
                << "("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0 and col("<<i<<") = 0" ;

#ifdef SPARSEMATRIX_CHECK
        if (i >= rowSize() || i >= colSize())
        {
            std::cerr << "ERROR: invalid write access to row and column "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        if (((Index)NL) != ((Index)NC) || nRow != nCol)
        {
            clearRow(i);
            clearCol(i);
        }
        else
        {
            // Here we assume the matrix is symmetric
            Index bi=0; split_row_index(i, bi);
            compress();
            Index rowId = i * (Index)rowIndex.size() / nBlocRow;
            if (sortedFind(rowIndex, i, rowId))
            {
                Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
                for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
                {
                    Bloc& b = colsValue[xj];
                    for (Index bj = 0; bj < NC; ++bj)
                        traits::v(b, bi, bj) = 0;
                    Index j = colsIndex[xj];

                    //Removed "if (j!=i)" commented as "non-diagonal bloc" optimization that was in fact a bug
                    //TODO : This whole block of code may then need refactoring

                    Bloc* bloc = wbloc(j,i,false);
                    if (bloc)
                    {
                        for (Index bj = 0; bj < NL; ++bj)
                            traits::v(*bloc, bj, bi) = 0;
                    }
                }
            }
        }
    }

    void clear()
    {
        for (Index i=0; i < (Index)colsValue.size(); ++i)
            traits::clear(colsValue[i]);
        compressed = colsValue.empty();
        btemp.clear();
    }

    /// @name Get information about the content and structure of this matrix (diagonal, band, sparse, full, block size, ...)
    /// @{

    /// @return type of elements stored in this matrix
    virtual ElementType getElementType() const { return traits::getElementType(); }

    /// @return size of elements stored in this matrix
    virtual std::size_t getElementSize() const { return sizeof(Real); }

    /// @return the category of this matrix
    virtual MatrixCategory getCategory() const { return MATRIX_SPARSE; }

    /// @return the number of rows in each block, or 1 of there are no fixed block size
    virtual Index getBlockRows() const { return NL; }

    /// @return the number of columns in each block, or 1 of there are no fixed block size
    virtual Index getBlockCols() const { return NC; }

    /// @return the number of rows of blocks
    virtual Index bRowSize() const { return rowBSize(); }

    /// @return the number of columns of blocks
    virtual Index bColSize() const { return colBSize(); }

    /// @return the width of the band on each side of the diagonal (only for band matrices)
    virtual Index getBandWidth() const { return NC-1; }

    /// @}

    /// @name Virtual iterator classes and methods
    /// @{

protected:
    virtual void bAccessorDelete(const InternalBlockAccessor* /*b*/) const {}
    virtual void bAccessorCopy(InternalBlockAccessor* /*b*/) const {}
    virtual SReal bAccessorElement(const InternalBlockAccessor* b, Index i, Index j) const
    {
        //return element(b->row * getBlockRows() + i, b->col * getBlockCols() + j);
        Index index = b->data;
        const Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        return (SReal)traits::v(data, i, j);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, Index i, Index j, double v)
    {
        //set(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
        Index index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        traits::v(data, i, j) = (Real)v;
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, Index i, Index j, double v)
    {
        Index index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        traits::v(data, i, j) += (Real)v;
    }

    template<class T>
    const T* bAccessorElementsCSRImpl(const InternalBlockAccessor* b, T* buffer) const
    {
        Index index = b->data;
        const Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                buffer[l*NC+c] = (T)traits::v(data, l, c);
        return buffer;
    }
    virtual const float* bAccessorElements(const InternalBlockAccessor* b, float* buffer) const
    {
        return bAccessorElementsCSRImpl<float>(b, buffer);
    }
    virtual const double* bAccessorElements(const InternalBlockAccessor* b, double* buffer) const
    {
        return bAccessorElementsCSRImpl<double>(b, buffer);
    }
    virtual const int* bAccessorElements(const InternalBlockAccessor* b, int* buffer) const
    {
        return bAccessorElementsCSRImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorSetCSRImpl(InternalBlockAccessor* b, const T* buffer)
    {
        Index index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                traits::v(data, l, c) = (Real)buffer[l*NC+c];
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const float* buffer)
    {
        bAccessorSetCSRImpl<float>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const double* buffer)
    {
        bAccessorSetCSRImpl<double>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const int* buffer)
    {
        bAccessorSetCSRImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorAddCSRImpl(InternalBlockAccessor* b, const T* buffer)
    {
        Index index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                traits::v(data, l, c) += (Real)buffer[l*NC+c];
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const float* buffer)
    {
        bAccessorAddCSRImpl<float>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const double* buffer)
    {
        bAccessorAddCSRImpl<double>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const int* buffer)
    {
        bAccessorAddCSRImpl<int>(b, buffer);
    }

public:

    /// Get read access to a bloc
    virtual BlockConstAccessor blocGet(Index i, Index j) const
    {
        ((Matrix*)this)->compress();

        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return createBlockConstAccessor(i, j, colId);
            }
        }
        return createBlockConstAccessor(-1-i, -1-j, (Index)0);
    }

    /// Get write access to a bloc
    virtual BlockAccessor blocGetW(Index i, Index j)
    {
        ((Matrix*)this)->compress();

        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return createBlockAccessor(i, j, colId);
            }
        }
        return createBlockAccessor(-1-i, -1-j, (Index)0);
    }

    /// Get write access to a bloc, possibly creating it
    virtual BlockAccessor blocCreate(Index i, Index j)
    {
        Index rowId = i * (Index)rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                dmsg_info_when(EMIT_EXTRA_MESSAGE)
                        << "("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<") found at "<<colId<<" (line "<<rowId<<")." ;
                return createBlockAccessor(i, j, colId);
            }
        }
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
                dmsg_info_when(EMIT_EXTRA_MESSAGE) << "("<<rowSize()<<","<<colSize()<<"): new temp bloc ("<<i<<","<<j<<")" ;
                btemp.push_back(IndexedBloc(i,j));
                traits::clear(btemp.back().value);
            }
            return createBlockAccessor(i, j, -(Index)btemp.size());
        }
    }

protected:
    virtual void itCopyColBlock(InternalColBlockIterator* /*it*/) const {}
    virtual void itDeleteColBlock(const InternalColBlockIterator* /*it*/) const {}
    virtual void itAccessColBlock(InternalColBlockIterator* it, BlockConstAccessor* b) const
    {
        Index index = it->data;
        setMatrix(b);
        getInternal(b)->row = it->row;
        getInternal(b)->data = index;
        getInternal(b)->col = colsIndex[index];
    }
    virtual void itIncColBlock(InternalColBlockIterator* it) const
    {
        Index index = it->data;
        ++index;
        it->data = index;
    }
    virtual void itDecColBlock(InternalColBlockIterator* it) const
    {
        Index index = it->data;
        --index;
        it->data = index;
    }
    virtual bool itEqColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        Index index = it->data;
        Index index2 = it2->data;
        return index == index2;
    }
    virtual bool itLessColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        Index index = it->data;
        Index index2 = it2->data;
        return index < index2;
    }

public:
    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(Index ib) const
    {
        ((Matrix*)this)->compress();
        Index rowId = ib * (Index)rowIndex.size() / nBlocRow;
        Index index = 0;
        if (sortedFind(rowIndex, ib, rowId))
        {
            index = rowBegin[rowId];
        }
        return createColBlockConstIterator(ib, index);
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(Index ib) const
    {
        ((Matrix*)this)->compress();
        Index rowId = ib * (Index)rowIndex.size() / nBlocRow;
        Index index2 = 0;
        if (sortedFind(rowIndex, ib, rowId))
        {
            index2 = rowBegin[rowId+1];
        }
        return createColBlockConstIterator(ib, index2);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(Index ib) const
    {
        ((Matrix*)this)->compress();
        Index rowId = ib * (Index)rowIndex.size() / nBlocRow;
        Index index = 0, index2 = 0;
        if (sortedFind(rowIndex, ib, rowId))
        {
            index = rowBegin[rowId];
            index2 = rowBegin[rowId+1];
        }
        return std::make_pair(createColBlockConstIterator(ib, index ),
                createColBlockConstIterator(ib, index2));
    }


protected:
    virtual void itCopyRowBlock(InternalRowBlockIterator* /*it*/) const {}
    virtual void itDeleteRowBlock(const InternalRowBlockIterator* /*it*/) const {}
    virtual Index itAccessRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        return rowIndex[rowId];
    }
    virtual ColBlockConstIterator itBeginRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        Index row = rowIndex[rowId];
        Index index = rowBegin[rowId];
        return createColBlockConstIterator(row, index);
    }
    virtual ColBlockConstIterator itEndRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        Index row = rowIndex[rowId];
        Index index2 = rowBegin[rowId+1];
        return createColBlockConstIterator(row, index2);
    }
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> itRangeRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        Index row = rowIndex[rowId];
        Index index = rowBegin[rowId];
        Index index2 = rowBegin[rowId+1];
        return std::make_pair(createColBlockConstIterator(row, index ),
                createColBlockConstIterator(row, index2));
    }

    virtual void itIncRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        ++rowId;
        it->data[0] = rowId;
    }
    virtual void itDecRowBlock(InternalRowBlockIterator* it) const
    {
        Index rowId = it->data[0];
        --rowId;
        it->data[0] = rowId;
    }
    virtual bool itEqRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        Index rowId = it->data[0];
        Index rowId2 = it2->data[0];
        return rowId == rowId2;
    }
    virtual bool itLessRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        Index rowId = it->data[0];
        Index rowId2 = it2->data[0];
        return rowId < rowId2;
    }

public:
    /// Get the iterator corresponding to the beginning of the rows of blocks
    virtual RowBlockConstIterator bRowsBegin() const
    {
        ((Matrix*)this)->compress();
        return createRowBlockConstIterator(0, 0);
    }

    /// Get the iterator corresponding to the end of the rows of blocks
    virtual RowBlockConstIterator bRowsEnd() const
    {
        ((Matrix*)this)->compress();
        return createRowBlockConstIterator((Index)rowIndex.size(), 0);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<RowBlockConstIterator, RowBlockConstIterator> bRowsRange() const
    {
        ((Matrix*)this)->compress();
        return std::make_pair(createRowBlockConstIterator(0, 0),
                createRowBlockConstIterator((Index)rowIndex.size(), 0));
    }

    /// @}

protected:


    /// @name setter/getter & product methods on template vector types
    /// @{

    template<class Vec> static Real vget(const Vec& vec, Index i, Index j, Index k) { return vget( vec, i*j+k ); }
    template<class Vec> static Real vget(const helper::vector<Vec>&vec, Index i, Index /*j*/, Index k) { return vec[i][k]; }

                          static Real  vget(const defaulttype::BaseVector& vec, Index i) { return vec.element(i); }
    template<class Real2> static Real2 vget(const FullVector<Real2>& vec, Index i) { return vec[i]; }


    template<class Vec> static void vset(Vec& vec, Index i, Index j, Index k, Real v) { vset( vec, i*j+k, v ); }
    template<class Vec> static void vset(helper::vector<Vec>&vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] = v; }

                          static void vset(defaulttype::BaseVector& vec, Index i, Real v) { vec.set(i, v); }
    template<class Real2> static void vset(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] = v; }


    template<class Vec> static void vadd(Vec& vec, Index i, Index j, Index k, Real v) { vadd( vec, i*j+k, v ); }
    template<class Vec> static void vadd(helper::vector<Vec>&vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] += v; }

                          static void vadd(defaulttype::BaseVector& vec, Index i, Real v) { vec.add(i, v); }
    template<class Real2> static void vadd(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] += v; }

    template<class Vec> static void vresize(Vec& vec, Index /*blockSize*/, Index totalSize) { vec.resize( totalSize ); }
    template<class Vec> static void vresize(helper::vector<Vec>&vec, Index blockSize, Index /*totalSize*/) { vec.resize( blockSize ); }



      /** Product of the matrix with a templated vector res = this * vec*/
      template<class Real2, class V1, class V2>
      void tmul(V1& res, const V2& vec) const
      {
          assert( vec.size()%bColSize() == 0 ); // vec.size() must be a multiple of block size.

          ((Matrix*)this)->compress();
          vresize( res, rowBSize(), rowSize() );
          for (Index xi = 0; xi < (Index)rowIndex.size(); ++xi)  // for each non-empty block row
          {
              defaulttype::Vec<NL,Real2> r;  // local block-sized vector to accumulate the product of the block row  with the large vector

              // multiply the non-null blocks with the corresponding chunks of the large vector
              Range rowRange(rowBegin[xi], rowBegin[xi+1]);
              for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
              {
                  // transfer a chunk of large vector to a local block-sized vector
                  defaulttype::Vec<NC,Real2> v;
                  //Index jN = colsIndex[xj] * NC;    // scalar column index
                  for (Index bj = 0; bj < NC; ++bj)
                      v[bj] = vget(vec,colsIndex[xj],NC,bj);

                  // multiply the block with the local vector
                  const Bloc& b = colsValue[xj];    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                  for (Index bi = 0; bi < NL; ++bi)
                      for (Index bj = 0; bj < NC; ++bj)
                          r[bi] += traits::v(b, bi, bj) * v[bj];
              }

              // transfer the local result  to the large result vector
              //Index iN = rowIndex[xi] * NL;                      // scalar row index
              for (Index bi = 0; bi < NL; ++bi)
                  vset(res, rowIndex[xi], NL, bi, r[bi]);
          }
      }


      /** Product of the matrix with a templated vector res += this * vec*/
      template<class Real2, class V1, class V2>
      void taddMul(V1& res, const V2& vec) const
      {
          assert( vec.size()%bColSize() == 0 ); // vec.size() must be a multiple of block size.

          ((Matrix*)this)->compress();
          vresize( res, rowBSize(), rowSize() );
          for (Index xi = 0; xi < (Index)rowIndex.size(); ++xi)  // for each non-empty block row
          {
              defaulttype::Vec<NL,Real2> r;  // local block-sized vector to accumulate the product of the block row  with the large vector

              // multiply the non-null blocks with the corresponding chunks of the large vector
              Range rowRange(rowBegin[xi], rowBegin[xi+1]);
              for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
              {
                  // transfer a chunk of large vector to a local block-sized vector
                  defaulttype::Vec<NC,Real2> v;
                  //Index jN = colsIndex[xj] * NC;    // scalar column index
                  for (Index bj = 0; bj < NC; ++bj)
                      v[bj] = vget(vec,colsIndex[xj],NC,bj);

                  // multiply the block with the local vector
                  const Bloc& b = colsValue[xj];    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                  for (Index bi = 0; bi < NL; ++bi)
                      for (Index bj = 0; bj < NC; ++bj)
                          r[bi] += traits::v(b, bi, bj) * v[bj];
              }

              // transfer the local result  to the large result vector
              //Index iN = rowIndex[xi] * NL;                      // scalar row index
              for (Index bi = 0; bi < NL; ++bi)
                  vadd(res, rowIndex[xi], NL, bi, r[bi]);
          }
      }


      /** Product of the matrix with a templated vector that have the size of the bloc res += this * [vec,...,vec]^T */
      template<class Real2, class V1, class V2>
      void taddMul_by_line(V1& res, const V2& vec) const
      {
          assert( vec.size() == NC ); // vec.size() must have the block size.

          ((Matrix*)this)->compress();
          vresize( res, rowBSize(), rowSize() );
          for (Index xi = 0; xi < (Index)rowIndex.size(); ++xi)  // for each non-empty block row
          {
              defaulttype::Vec<NL,Real2> r;  // local block-sized vector to accumulate the product of the block row  with the large vector

              // multiply the non-null blocks with the corresponding chunks of the large vector
              Range rowRange(rowBegin[xi], rowBegin[xi+1]);
              for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
              {
                  // multiply the block with the local vector
                  const Bloc& b = colsValue[xj];    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                  for (Index bi = 0; bi < NL; ++bi)
                      for (Index bj = 0; bj < NC; ++bj)
                          r[bi] += traits::v(b, bi, bj) * vec[bj];
              }

              // transfer the local result  to the large result vector
              //Index iN = rowIndex[xi] * NL;                      // scalar row index
              for (Index bi = 0; bi < NL; ++bi)
                  vadd(res, rowIndex[xi], NL, bi, r[bi]);
          }
      }

      /** Product of the transpose with a templated vector and add it to res   res += this^T * vec */
      template<class Real2, class V1, class V2>
      void taddMulTranspose(V1& res, const V2& vec) const
      {
          assert( vec.size()%bRowSize() == 0 ); // vec.size() must be a multiple of block size.

          ((Matrix*)this)->compress();
          vresize( res, colBSize(), colSize() );
          for (Index xi = 0; xi < rowIndex.size(); ++xi) // for each non-empty block row (i.e. column of the transpose)
          {
              // copy the corresponding chunk of the input to a local vector
              defaulttype::Vec<NL,Real2> v;
              //Index iN = rowIndex[xi] * NL;    // index of the row in the vector
              for (Index bi = 0; bi < NL; ++bi)
                  v[bi] = vget(vec, rowIndex[xi], NL, bi);

              // accumulate the product of the column with the local vector
              Range rowRange(rowBegin[xi], rowBegin[xi+1]);
              for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj) // for each non-empty block in the row
              {
                  const Bloc& b = colsValue[xj]; // non-empty block

                  defaulttype::Vec<NC,Real2> r;  // local vector to store the product
                  //Index jN = colsIndex[xj] * NC;

                  // columnwise bloc-vector product
                  for (Index bj = 0; bj < NC; ++bj)
                      r[bj] = traits::v(b, 0, bj) * v[0];
                  for (Index bi = 1; bi < NL; ++bi)
                      for (Index bj = 0; bj < NC; ++bj)
                          r[bj] += traits::v(b, bi, bj) * v[bi];

                  // accumulate the product to the result
                  for (Index bj = 0; bj < NC; ++bj)
                      vadd(res, colsIndex[xj], NC, bj, r[bj]);
              }
          }
      }


/// @}


public:


      /// @name Matrix operators
      /// @{


    /** Compute res = this * m
      @warning The block sizes must be compatible, i.e. this::NC==m::NR and res::NR==this::NR and res::NC==m::NC.
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RVB, typename RVI, typename MB, typename MVB, typename MVI >
    void mul( CompressedRowSparseMatrix<RB,RVB,RVI>& res, const CompressedRowSparseMatrix<MB,MVB,MVI>& m ) const
    {
        assert( Bloc::nbCols == MB::nbLines );
        assert( RB::nbLines == Bloc::nbLines );
        assert( MB::nbCols == RB::nbCols );

        assert( colSize() == m.rowSize() );

        ((Matrix*)this)->compress();  /// \warning this violates the const-ness of the method
        ((CompressedRowSparseMatrix<MB,MVB,MVI>*)&m)->compress();  /// \warning this violates the const-ness of the parameter


        res.resize( this->nRow, m.nCol );  // clear and resize the result

        if( m.rowIndex.empty() ) return; // if m is null

        for( Index xi = 0; xi < (Index)rowIndex.size(); ++xi )  // for each non-null block row
        {
            unsigned mr = 0; // block row index in m

            Index row = rowIndex[xi];      // block row

            Range rowRange( rowBegin[xi], rowBegin[xi+1] );
            for( Index xj = rowRange.begin() ; xj < rowRange.end() ; ++xj )  // for each non-null block
            {
                Index col = colsIndex[xj];     // block column
                const Bloc& b = colsValue[xj]; // block value

                // find the non-null row in m, if any
                while( mr<m.rowIndex.size() && m.rowIndex[mr]<col ) mr++;
                if( mr==m.rowIndex.size() || m.rowIndex[mr] > col ) continue;  // no matching row, ignore this block

                // Accumulate  res[row] += b * m[col]
                Range mrowRange( m.rowBegin[mr], m.rowBegin[mr+1] );
                for( Index mj = mrowRange.begin() ; mj< mrowRange.end() ; ++mj ) // for each non-null block in  m[col]
                {
                    Index mcol = m.colsIndex[mj];     // column index of the non-null block
                    *res.wbloc(row,mcol,true) += b * m.colsValue[mj];  // find the matching bloc in res, and accumulate the block product
                }
            }
        }
        res.compress();
    }


    /** Compute res = this.transpose * m
      @warning The block sizes must be compatible, i.e. this::NR==m::NR and res::NR==this::NC and res::NC==m::NC
      The basic algorithm consists in accumulating rows of m to rows of res: foreach row { foreach col { res[row] += this[row,col] * m[col] } }
      @warning matrices this and m must be compressed
      */
    template<typename RB, typename RVB, typename RVI, typename MB, typename MVB, typename MVI >
    void mulTranspose( CompressedRowSparseMatrix<RB,RVB,RVI>& res, const CompressedRowSparseMatrix<MB,MVB,MVI>& m ) const
    {
        assert( Bloc::nbLines == MB::nbLines );
        assert( RB::nbLines == Bloc::nbCols );
        assert( MB::nbCols == RB::nbCols );

        assert( rowSize() == m.rowSize() );

        // must already be compressed, since matrices are const they cannot be modified
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
                    *res.wbloc(row,mcol,true) += b.multTranspose( m.colsValue[mj] );  // find the matching bloc in res, and accumulate the block product
                }
            }
        }
        res.compress();
    }



    /** @returns this + m
      @warning The block must be the same (same type and same size)
      @warning The matrices must have the same mathematical size
      @warning matrices this and m must be compressed
      */
    CompressedRowSparseMatrix<TBloc,TVecBloc,TVecIndex> operator+( const CompressedRowSparseMatrix<TBloc,TVecBloc,TVecIndex>& m ) const
    {
        CompressedRowSparseMatrix<TBloc,TVecBloc,TVecIndex> res = *this;
        res += m;
        return res;
    }
    /// @}

    /// @name specialization of product methods on a few vector types
    /// @{

    /// equal result = this * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template< typename V1, typename V2 >
    void mul( V2& result, const V1& v ) const
    {
        tmul< Real, V2, V1 >(result, v);
    }


    /// equal result += this^T * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template< typename V1, typename V2 >
    void addMultTranspose( V1& result, const V2& v ) const
    {
        taddMulTranspose< Real, V1, V2 >(result, v);
    }

    /// @returns this * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template<class Vec>
    Vec operator*(const Vec& v) const
    {
        Vec res;
        mul( res, v );
        return res;
    }


    /// result += this * (v,...,v)^T
    /// v has the size of one bloc
    template< typename V, typename Real2 >
    void addMul_by_line( V& res, const defaulttype::Vec<NC,Real2>& v ) const
    {
        taddMul_by_line< Real2,V,defaulttype::Vec<NC,Real2> >( res, v );
    }
    template< typename Real, typename V, typename V2 >
    void addMul_by_line( V& res, const V2& v ) const
    {
        taddMul_by_line< Real,V,V2 >( res, v );
    }


    /// result += this * v
    template< typename V1, typename V2 >
    void addMul( V1& res, const V2& v ) const
    {
        taddMul< Real,V1,V2 >( res, v );
    }



    /// @}


    // methods for MatrixExpr support

    template<class M2>
    bool hasRef(const M2* m) const
    {
        return (const void*)this == (const void*)m;
    }

    std::string expr() const
    {
        return std::string(Name());
    }

    bool valid() const
    {
        return true;
    }


    /// dest += this
    /// different bloc types possible
    /// @todo how to optimize when same bloc types
    template<class Dest>
    void addTo(Dest* dest) const
    {
        for (Index xi = 0; xi < (Index)rowIndex.size(); ++xi)
        {
            Index iN = rowIndex[xi] * NL;
            Range rowRange(rowBegin[xi], rowBegin[xi+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index jN = colsIndex[xj] * NC;
                const Bloc& b = colsValue[xj];
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        dest->add(iN+bi, jN+bj, traits::v(b, bi, bj));
            }
        }
        if (!btemp.empty())
        {
            for (typename VecIndexedBloc::const_iterator it = btemp.begin(), itend = btemp.end(); it != itend; ++it)
            {
                Index iN = it->l * NL;
                Index jN = it->c * NC;
                const Bloc& b = it->value;
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        dest->add(iN+bi, jN+bj, traits::v(b, bi, bj));
            }
        }
    }

protected:

    /// add ? this += m : this = m
    /// m can be the same as this
    template<class M>
    void equal( const M& m, bool add = false )
    {
        if (m.hasRef(this))
        {
            Matrix tmp;
            tmp.resize(m.rowSize(), m.colSize());
            m.addTo(&tmp);
            if (add)
                tmp.addTo(this);
            else
                swap(tmp);
        }
        else
        {
            if (!add)
                resize(m.rowSize(), m.colSize());
            m.addTo(this);
        }
    }

    /// this += m
    template<class M>
    inline void addEqual( const M& m )
    {
        equal( m, true );
    }



public:

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        if (&m == this) return;
        resize(m.rowSize(), m.colSize());
        m.addTo(this);
    }

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator+=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        addEqual(m);
    }

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator-=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        equal(MatrixExpr< MatrixNegative< CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2> > >(MatrixNegative< CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2> >(m)), true);
    }

    template<class Expr2>
    void operator=(const MatrixExpr< Expr2 >& m)
    {
        equal(m, false);
    }

    template<class Expr2>
    void operator+=(const MatrixExpr< Expr2 >& m)
    {
        addEqual(m);
    }

    template<class Expr2>
    void operator-=(const MatrixExpr< Expr2 >& m)
    {
        addEqual(MatrixExpr< MatrixNegative< Expr2 > >(MatrixNegative< Expr2 >(m)));
    }

    MatrixExpr< MatrixTranspose< Matrix > > t() const
    {
        return MatrixExpr< MatrixTranspose< Matrix > >(MatrixTranspose< Matrix >(*this));
    }

    MatrixExpr< MatrixInverse< Matrix > > i() const
    {
        return MatrixExpr< MatrixInverse< Matrix > >(MatrixInverse< Matrix >(*this));
    }

    MatrixExpr< MatrixNegative< Matrix > > operator-() const
    {
        return MatrixExpr< MatrixNegative< Matrix > >(MatrixNegative< Matrix >(*this));
    }

    MatrixExpr< MatrixScale< Matrix, double > > operator*(const double& r) const
    {
        return MatrixExpr< MatrixScale< Matrix, double > >(MatrixScale< Matrix, double >(*this, r));
    }


    static const char* Name()
    {
        static std::string name = std::string("CompressedRowSparseMatrix") + std::string(traits::Name());
        return name.c_str();
    }

    bool check_matrix()
    {
        return check_matrix(
                this->getColsValue().size(),
                this->rowBSize(),
                this->colBSize(),
                (Index *) &(this->getRowBegin()[0]),
                (Index *) &(this->getColsIndex()[0]),
                (double *) &(this->getColsValue()[0])
                );
    }

    static bool check_matrix(
        Index nzmax,// nb values
        Index m,// number of row
        Index n,// number of columns
        Index * a_p,// column pointers (size n+1) or col indices (size nzmax)
        Index * a_i,// row indices, size nzmax
        double * a_x// numerical values, size nzmax
    )
    {
        // check ap, size m beecause ther is at least the diagonal value wich is different of 0
        if (a_p[0]!=0)
        {
            msg_error("CompressedRowSparseMatrix") << "First value of row indices (a_p) should be 0" ;
            return false;
        }

        for (Index i=1; i<=m; i++)
        {
            if (a_p[i]<=a_p[i-1])
            {
                msg_error("CompressedRowSparseMatrix") << "Row (a_p) indices are not sorted index " << i-1 << " : " << a_p[i-1] << " , " << i << " : " << a_p[i] ;
                return false;
            }
        }
        if (nzmax == -1)
        {
            nzmax = a_p[m];
        }
        else if (a_p[m]!=nzmax)
        {
            msg_error("CompressedRowSparseMatrix") << "Last value of row indices (a_p) should be " << nzmax << " and is " << a_p[m] ;
            return false;
        }


        Index k=1;
        for (Index i=0; i<nzmax; i++)
        {
            i++;
            for (; i<a_p[k]; i++)
            {
                if (a_i[i] <= a_i[i-1])
                {
                    msg_error("CompressedRowSparseMatrix") << "Column (a_i) indices are not sorted index " << i-1 << " : " << a_i[i-1] << " , " << i << " : " << a_p[i] ;
                    return false;
                }
                if (a_i[i]<0 || a_i[i]>=n)
                {
                    msg_error("CompressedRowSparseMatrix") << "Column (a_i) indices are not correct " << i << " : " << a_i[i] ;
                    return false;
                }
            }
            k++;
        }

        for (Index i=0; i<nzmax; i++)
        {
            if (a_x[i]==0)
            {
                msg_warning("CompressedRowSparseMatrix") << "Matrix contains 0 , index " << i ;
                return false;
            }
        }

        if (n!=m)
        {
            msg_error("CompressedRowSparseMatrix") << "The matrix is not square" ;
            return false;
        }

        msg_info("CompressedRowSparseMatrix") << "Check_matrix passed successfully" ;
        return true;
    }
};

#ifdef SPARSEMATRIX_CHECK
#undef SPARSEMATRIX_CHECK
#endif
#ifdef SPARSEMATRIX_VERBOSE
#undef SPARSEMATRIX_VERBOSE
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
