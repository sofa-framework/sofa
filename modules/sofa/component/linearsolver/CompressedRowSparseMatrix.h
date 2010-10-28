/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/component/linearsolver/MatrixExpr.h>
#include <sofa/component/linearsolver/matrix_bloc_traits.h>
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

template<typename TBloc, typename TVecBloc = helper::vector<TBloc>, typename TVecIndex = helper::vector<int> >
class CompressedRowSparseMatrix : public defaulttype::BaseMatrix
{
public:
    typedef CompressedRowSparseMatrix<TBloc,TVecBloc,TVecIndex> Matrix;
    typedef TBloc Bloc;
    typedef matrix_bloc_traits<Bloc> traits;
    typedef typename traits::Real Real;
    enum { NL = traits::NL };
    enum { NC = traits::NC };
    typedef int Index;

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

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

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
        return sortedFind(v, Range(0,v.size()), val, result);
    }

public :
    Index nRow,nCol;
    Index nBlocRow,nBlocCol;
    bool compressed;
    VecIndex rowIndex;
    VecIndex rowBegin;
    VecIndex colsIndex;
    VecBloc  colsValue;
    VecIndexedBloc btemp;

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

    CompressedRowSparseMatrix(int nbRow, int nbCol)
        : nRow(nbRow), nCol(nbCol),
          nBlocRow((nbRow + NL-1) / NL), nBlocCol((nbCol + NC-1) / NC),
          compressed(true)
    {
    }

    unsigned int rowBSize() const
    {
        return nBlocRow;
    }

    unsigned int colBSize() const
    {
        return nBlocCol;
    }

    const VecIndex& getRowIndex() const { return rowIndex; }
    const VecIndex& getRowBegin() const { return rowBegin; }
    Range getRowRange(int id) const { return Range(rowBegin[id], rowBegin[id+1]); }
    const VecIndex& getColsIndex() const { return colsIndex; }
    const VecBloc& getColsValue() const { return colsValue; }

    void resizeBloc(int nbBRow, int nbBCol)
    {
        if (nBlocRow == nbBRow && nBlocRow == nbBCol)
        {
            // just clear the matrix
            for (unsigned int i=0; i < colsValue.size(); ++i)
                traits::clear(colsValue[i]);
            compressed = colsValue.empty();
            btemp.clear();
        }
        else
        {
#ifdef SPARSEMATRIX_VERBOSE
            std::cout << /* this->Name()  <<  */": resize("<<nbBRow<<"*"<<NL<<","<<nbBCol<<"*"<<NC<<")"<<std::endl;
#endif
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

    void compress()
    {
        if (compressed && btemp.empty()) return;
        if (!btemp.empty())
        {
#ifdef SPARSEMATRIX_VERBOSE
            std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): sort "<<btemp.size()<<" temp blocs."<<std::endl;
#endif
            std::sort(btemp.begin(),btemp.end());
#ifdef SPARSEMATRIX_VERBOSE
            std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): blocs sorted."<<std::endl;
#endif
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
        const Index oldNRow = oldRowIndex.size();
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
#ifdef SPARSEMATRIX_VERBOSE
            std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): inRowIndex = "<<inRowIndex<<" , bRowIndex = "<<bRowIndex<<""<<std::endl;
#endif
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
                //colsIndex.insert(colsIndex.end(), inRow.begin(oldColsIndex), inRow.end(oldColsIndex));
                //colsValue.insert(colsValue.end(), inRow.begin(oldColsValue), inRow.end(oldColsValue));
                //outValId += inRow.size();
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
//#ifdef SPARSEMATRIX_VERBOSE
        //          std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): compressed " << oldColsIndex.size()<<" old blocs and " << btemp.size() << " temp blocs into " << rowIndex.size() << " lines and " << colsIndex.size() << " blocs."<<std::endl;
//#endif
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
        if ((int)rowIndex.size() >= nRow) return;
        oldRowIndex.swap(rowIndex);
        oldRowBegin.swap(rowBegin);
        rowIndex.resize(nRow);
        rowBegin.resize(nRow+1);
        for (int i=0; i<nRow; ++i) rowIndex[i] = i;
        int j = 0;
        int b = 0;
        for (unsigned int i=0; i<oldRowIndex.size(); ++i)
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
        int ndiag = 0;
        for (unsigned int r=0; r<rowIndex.size(); ++r)
        {
            int i = rowIndex[r];
            int b = rowBegin[r];
            int e = rowBegin[r+1];
            int t = b;
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
        int nv = 0;
        for (int i=0; i<nRow; ++i) rowIndex[i] = i;
        int j = 0;
        for (unsigned int i=0; i<oldRowIndex.size(); ++i)
        {
            for (; j<oldRowIndex[i]; ++j)
            {
                rowBegin[j] = nv;
                colsIndex[nv] = j;
                traits::clear(colsValue[nv]);
                ++nv;
            }
            rowBegin[j] = nv;
            int b = oldRowBegin[i];
            int e = oldRowBegin[i+1];
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
    void shiftIndices(int base)
    {
        for (unsigned int i=0; i<rowIndex.size(); ++i)
            rowIndex[i] += base;
        for (unsigned int i=0; i<rowBegin.size(); ++i)
            rowBegin[i] += base;
        for (unsigned int i=0; i<colsIndex.size(); ++i)
            colsIndex[i] += base;
    }

    // filtering-out part of a matrix
    typedef bool filter_fn    (int   i  , int   j  , Bloc& val, const Bloc&   ref  );
    static bool       nonzeros(int /*i*/, int /*j*/, Bloc& val, const Bloc& /*ref*/) { return (!traits::empty(val)); }
    static bool       nonsmall(int /*i*/, int /*j*/, Bloc& val, const Bloc&   ref  )
    {
        for (int bi = 0; bi < NL; ++bi)
            for (int bj = 0; bj < NC; ++bj)
                if (helper::rabs(traits::v(val, bi, bj)) >= ref) return true;
        return false;
    }
    static bool upper         (int   i  , int   j  , Bloc& val, const Bloc& /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (int bi = 1; bi < NL; ++bi)
                for (int bj = 0; bj < bi; ++bj)
                    traits::v(val, bi, bj) = 0;
        }
        return i*NL <= j*NC;
    }
    static bool lower         (int   i  , int   j  , Bloc& val, const Bloc& /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (int bi = 0; bi < NL-1; ++bi)
                for (int bj = bi+1; bj < NC; ++bj)
                    traits::v(val, bi, bj) = 0;
        }
        return i*NL >= j*NC;
    }
    static bool upper_nonzeros(int   i  , int   j  , Bloc& val, const Bloc&   ref  ) { return upper(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool lower_nonzeros(int   i  , int   j  , Bloc& val, const Bloc&   ref  ) { return lower(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool upper_nonsmall(int   i  , int   j  , Bloc& val, const Bloc&   ref  ) { return upper(i,j,val,ref) && nonsmall(i,j,val,ref); }
    static bool lower_nonsmall(int   i  , int   j  , Bloc& val, const Bloc&   ref  ) { return lower(i,j,val,ref) && nonsmall(i,j,val,ref); }

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

        int vid = 0;
        for (unsigned int rowId = 0; rowId < M.rowIndex.size(); ++rowId)
        {
            int i = M.rowIndex[rowId];
            rowIndex.push_back(i);
            rowBegin.push_back(vid);
            Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);
            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                int j = M.colsIndex[xj];
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
        filterValues(M, nonzeros);
    }

    void copyNonSmall(Matrix& M, const Bloc& ref)
    {
        filterValues(M, nonzeros, ref);
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

    void copyLowerNonZeros(Matrix& M)
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

    const Bloc& bloc(int i, int j) const
    {
        static Bloc empty;
        int rowId = i * rowIndex.size() / nBlocRow;
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

    Bloc* wbloc(int i, int j, bool create = false)
    {
        int rowId = i * rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            int colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
#ifdef SPARSEMATRIX_VERBOSE
                std::cout << /* this->Name()  <<  */"("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<") found at "<<colId<<" (line "<<rowId<<")."<<std::endl;
#endif
                return &colsValue[colId];
            }
        }
        if (create)
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
#ifdef SPARSEMATRIX_VERBOSE
                std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): new temp bloc ("<<i<<","<<j<<")"<<std::endl;
#endif
                btemp.push_back(IndexedBloc(i,j));
                traits::clear(btemp.back().value);
            }
            return &btemp.back().value;
        }
        return NULL;
    }

    unsigned int rowSize() const
    {
        return nRow;
    }

    unsigned int colSize() const
    {
        return nCol;
    }

    void resize(int nbRow, int nbCol)
    {
#ifdef SPARSEMATRIX_VERBOSE
        if (nbRow != (int)rowSize() || nbCol != (int)colSize())
            std::cout << /* this->Name()  <<  */": resize("<<nbRow<<","<<nbCol<<")"<<std::endl;
#endif
        resizeBloc((nbRow + NL-1) / NL, (nbCol + NC-1) / NC);
        nRow = nbRow;
        nCol = nbCol;
    }

    SReal element(int i, int j) const
    {
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        int bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        ((Matrix*)this)->compress();
        return traits::v(bloc(i, j), bi, bj);
    }

    void set(int i, int j, double v)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        int bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<")["<<bi<<","<<bj<<"] = "<<v<<std::endl;
#endif
        traits::v(*wbloc(i,j,true), bi, bj) = (Real)v;
    }

    void add(int i, int j, double v)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        int bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<")["<<bi<<","<<bj<<"] += "<<v<<std::endl;
#endif
        traits::v(*wbloc(i,j,true), bi, bj) += (Real)v;
    }

    void clear(int i, int j)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        int bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        compress();
        Bloc* b = wbloc(i,j,false);
        if (b)
            traits::v(*b, bi, bj) = 0;
    }

    void clearRow(int i)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize())
        {
            std::cerr << "ERROR: invalid write access to row "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        int bi=0; split_row_index(i, bi);
        compress();
        /*
        for (int j=0; j<nBlocCol; ++j)
        {
            Bloc* b = wbloc(i,j,false);
            if (b)
            {
                for (int bj = 0; bj < NC; ++bj)
                    traits::v(*b, bi, bj) = 0;
            }
        }
        */
        int rowId = i * rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Bloc& b = colsValue[xj];
                for (int bj = 0; bj < NC; ++bj)
                    traits::v(b, bi, bj) = 0;
            }
        }
    }

    void clearCol(int j)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): col("<<j<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)j >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to column "<<j<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        int bj=0; split_col_index(j, bj);
        compress();
        for (int i=0; i<nBlocRow; ++i)
        {
            Bloc* b = wbloc(i,j,false);
            if (b)
            {
                for (int bi = 0; bi < NL; ++bi)
                    traits::v(*b, bi, bj) = 0;
            }
        }
    }

    void clearRowCol(int i)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0 and col("<<i<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)i >= (unsigned)colSize())
        {
            std::cerr << "ERROR: invalid write access to row and column "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        if ((int)NL != (int)NC || nRow != nCol)
        {
            clearRow(i);
            clearCol(i);
        }
        else
        {
            //std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): sparse row("<<i<<") = 0 and col("<<i<<") = 0"<<std::endl;
            // Here we assume the matrix is symmetric
            int bi=0; split_row_index(i, bi);
            compress();
            int rowId = i * rowIndex.size() / nBlocRow;
            if (sortedFind(rowIndex, i, rowId))
            {
                Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
                for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
                {
                    Bloc& b = colsValue[xj];
                    for (int bj = 0; bj < NC; ++bj)
                        traits::v(b, bi, bj) = 0;
                    int j = colsIndex[xj];
                    if (j != i)
                    {
                        // non diagonal bloc
                        Bloc* b = wbloc(j,i,false);
                        if (b)
                        {
                            for (int bj = 0; bj < NL; ++bj)
                                traits::v(*b, bj, bi) = 0;
                        }
                    }
                }
            }
        }
    }

    void clear()
    {
        for (unsigned int i=0; i < colsValue.size(); ++i)
            traits::clear(colsValue[i]);
        compressed = colsValue.empty();
        btemp.clear();
    }

    /// @name Get information about the content and structure of this matrix (diagonal, band, sparse, full, block size, ...)
    /// @{

    /// @return type of elements stored in this matrix
    virtual ElementType getElementType() const { return traits::getElementType(); }

    /// @return size of elements stored in this matrix
    virtual unsigned int getElementSize() const { return sizeof(Real); }

    /// @return the category of this matrix
    virtual MatrixCategory getCategory() const { return MATRIX_SPARSE; }

    /// @return the number of rows in each block, or 1 of there are no fixed block size
    virtual int getBlockRows() const { return NL; }

    /// @return the number of columns in each block, or 1 of there are no fixed block size
    virtual int getBlockCols() const { return NC; }

    /// @return the number of rows of blocks
    virtual int bRowSize() const { return rowBSize(); }

    /// @return the number of columns of blocks
    virtual int bColSize() const { return colBSize(); }

    /// @return the width of the band on each side of the diagonal (only for band matrices)
    virtual int getBandWidth() const { return NC-1; }

    /// @}

    /// @name Virtual iterator classes and methods
    /// @{

protected:
    virtual void bAccessorDelete(const InternalBlockAccessor* /*b*/) const {}
    virtual void bAccessorCopy(InternalBlockAccessor* /*b*/) const {}
    virtual SReal bAccessorElement(const InternalBlockAccessor* b, int i, int j) const
    {
        //return element(b->row * getBlockRows() + i, b->col * getBlockCols() + j);
        int index = b->data;
        const Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        return (SReal)traits::v(data, i, j);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, int i, int j, double v)
    {
        //set(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
        int index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        traits::v(data, i, j) = (Real)v;
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, int i, int j, double v)
    {
        //add(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
        int index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        traits::v(data, i, j) += (Real)v;
    }

    template<class T>
    const T* bAccessorElementsCSRImpl(const InternalBlockAccessor* b, T* buffer) const
    {
        int index = b->data;
        const Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
        int index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
        int index = b->data;
        Bloc& data = (index >= 0) ? colsValue[index] : btemp[-index-1].value;
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
    virtual BlockConstAccessor blocGet(int i, int j) const
    {
        ((Matrix*)this)->compress();

        int rowId = i * rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return createBlockConstAccessor(i, j, colId);
            }
        }
        return createBlockConstAccessor(-1-i, -1-j, 0);
    }

    /// Get write access to a bloc
    virtual BlockAccessor blocGetW(int i, int j)
    {
        ((Matrix*)this)->compress();

        int rowId = i * rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
                return createBlockAccessor(i, j, colId);
            }
        }
        return createBlockAccessor(-1-i, -1-j, 0);
    }

    /// Get write access to a bloc, possibly creating it
    virtual BlockAccessor blocCreate(int i, int j)
    {
        int rowId = i * rowIndex.size() / nBlocRow;
        if (sortedFind(rowIndex, i, rowId))
        {
            Range rowRange(rowBegin[rowId], rowBegin[rowId+1]);
            int colId = rowRange.begin() + j * rowRange.size() / nBlocCol;
            if (sortedFind(colsIndex, rowRange, j, colId))
            {
#ifdef SPARSEMATRIX_VERBOSE
                std::cout << /* this->Name()  <<  */"("<<rowBSize()<<"*"<<NL<<","<<colBSize()<<"*"<<NC<<"): bloc("<<i<<","<<j<<") found at "<<colId<<" (line "<<rowId<<")."<<std::endl;
#endif
                return createBlockAccessor(i, j, colId);
            }
        }
        //if (create)
        {
            if (btemp.empty() || btemp.back().l != i || btemp.back().c != j)
            {
#ifdef SPARSEMATRIX_VERBOSE
                std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): new temp bloc ("<<i<<","<<j<<")"<<std::endl;
#endif
                btemp.push_back(IndexedBloc(i,j));
                traits::clear(btemp.back().value);
            }
            return createBlockAccessor(i, j, -(int)btemp.size());
        }
    }

protected:
    virtual void itCopyColBlock(InternalColBlockIterator* /*it*/) const {}
    virtual void itDeleteColBlock(const InternalColBlockIterator* /*it*/) const {}
    virtual void itAccessColBlock(InternalColBlockIterator* it, BlockConstAccessor* b) const
    {
        int index = it->data;
        setMatrix(b);
        getInternal(b)->row = it->row;
        getInternal(b)->data = index;
        getInternal(b)->col = colsIndex[index];
    }
    virtual void itIncColBlock(InternalColBlockIterator* it) const
    {
        int index = it->data;
        ++index;
        it->data = index;
    }
    virtual void itDecColBlock(InternalColBlockIterator* it) const
    {
        int index = it->data;
        --index;
        it->data = index;
    }
    virtual bool itEqColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        int index = it->data;
        int index2 = it2->data;
        return index == index2;
    }
    virtual bool itLessColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        int index = it->data;
        int index2 = it2->data;
        return index < index2;
    }

public:
    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(int ib) const
    {
        ((Matrix*)this)->compress();
        int rowId = ib * rowIndex.size() / nBlocRow;
        int index = 0;
        if (sortedFind(rowIndex, ib, rowId))
        {
            index = rowBegin[rowId];
        }
        return createColBlockConstIterator(ib, index);
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(int ib) const
    {
        ((Matrix*)this)->compress();
        int rowId = ib * rowIndex.size() / nBlocRow;
        int index2 = 0;
        if (sortedFind(rowIndex, ib, rowId))
        {
            index2 = rowBegin[rowId+1];
        }
        return createColBlockConstIterator(ib, index2);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(int ib) const
    {
        ((Matrix*)this)->compress();
        int rowId = ib * rowIndex.size() / nBlocRow;
        int index = 0, index2 = 0;
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
    virtual int itAccessRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        return rowIndex[rowId];
    }
    virtual ColBlockConstIterator itBeginRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        int row = rowIndex[rowId];
        int index = rowBegin[rowId];
        return createColBlockConstIterator(row, index);
    }
    virtual ColBlockConstIterator itEndRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        int row = rowIndex[rowId];
        int index2 = rowBegin[rowId+1];
        return createColBlockConstIterator(row, index2);
    }
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> itRangeRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        int row = rowIndex[rowId];
        int index = rowBegin[rowId];
        int index2 = rowBegin[rowId+1];
        return std::make_pair(createColBlockConstIterator(row, index ),
                createColBlockConstIterator(row, index2));
    }

    virtual void itIncRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        ++rowId;
        it->data[0] = rowId;
    }
    virtual void itDecRowBlock(InternalRowBlockIterator* it) const
    {
        int rowId = it->data[0];
        --rowId;
        it->data[0] = rowId;
    }
    virtual bool itEqRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        int rowId = it->data[0];
        int rowId2 = it2->data[0];
        return rowId == rowId2;
    }
    virtual bool itLessRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        int rowId = it->data[0];
        int rowId2 = it2->data[0];
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
        return createRowBlockConstIterator(rowIndex.size(), 0);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<RowBlockConstIterator, RowBlockConstIterator> bRowsRange() const
    {
        ((Matrix*)this)->compress();
        return std::make_pair(createRowBlockConstIterator(0, 0),
                createRowBlockConstIterator(rowIndex.size(), 0));
    }

    /// @}

protected:
    template<class Real2>
    static Real vget(const defaulttype::BaseVector& vec, int i) { return vec.element(i); }
    template<class Real2> static Real2 vget(const FullVector<Real2>& vec, int i) { return vec[i]; }
    static void vset(defaulttype::BaseVector& vec, int i, Real v) { vec.set(i, v); }
    template<class Real2> static void vset(FullVector<Real2>& vec, int i, Real2 v) { vec[i] = v; }
    static void vadd(defaulttype::BaseVector& vec, int i, Real v) { vec.add(i, v); }
    template<class Real2> static void vadd(FullVector<Real2>& vec, int i, Real2 v) { vec[i] += v; }
public:

    template<class Real2, class V1, class V2>
    void tmul(V1& res, const V2& vec) const
    {
        ((Matrix*)this)->compress();
        res.resize(rowSize());
        for (unsigned int xi = 0; xi < rowIndex.size(); ++xi)
        {
            Index iN = rowIndex[xi] * NL;
            Range rowRange(rowBegin[xi], rowBegin[xi+1]);
            defaulttype::Vec<NL,Real2> r;
            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index jN = colsIndex[xj] * NC;
                const Bloc& b = colsValue[xj];
                defaulttype::Vec<NC,Real2> v;
                for (int bj = 0; bj < NC; ++bj)
                    v[bj] = vget(vec,jN + bj);
                for (int bi = 0; bi < NL; ++bi)
                    for (int bj = 0; bj < NC; ++bj)
                        r[bi] += traits::v(b, bi, bj) * v[bj];
            }
            for (int bi = 0; bi < NL; ++bi)
                vset(res, iN + bi, r[bi]);
        }
    }

    template<class Real2, class V1, class V2>
    void tmulTranspose(V1& res, const V2& vec) const
    {
        ((Matrix*)this)->compress();
        res.resize(colSize());
        for (unsigned int xi = 0; xi < rowIndex.size(); ++xi)
        {
            Index iN = rowIndex[xi] * NL;
            Range rowRange(rowBegin[xi], rowBegin[xi+1]);
            defaulttype::Vec<NL,Real2> v;
            for (int bi = 0; bi < NL; ++bi)
                v[bi] = vget(vec, iN + bi);
            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index jN = colsIndex[xj] * NC;
                const Bloc& b = colsValue[xj];
                defaulttype::Vec<NC,Real2> r;
                for (int bj = 0; bj < NC; ++bj)
                    r[bj] = traits::v(b, 0, bj) * v[0];
                for (int bi = 1; bi < NL; ++bi)
                    for (int bj = 0; bj < NC; ++bj)
                        r[bj] += traits::v(b, bi, bj) * v[bi];
                for (int bj = 0; bj < NC; ++bj)
                    vadd(res, jN + bj, r[bj]);
            }
        }
    }

    template<class Real2>
    void mul(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        tmul< Real2, FullVector<Real2>, FullVector<Real2> >(res, v);
    }

    template<class Real2>
    void mulTranspose(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        tmulTranspose< Real2, FullVector<Real2>, FullVector<Real2> >(res, v);
    }

    template<class Real2>
    void mul(FullVector<Real2>& res, const defaulttype::BaseVector* v) const
    {
        tmul< Real2, FullVector<Real2>, defaulttype::BaseVector >(res, *v);
    }

    template<class Real2>
    void mulTranspose(FullVector<Real2>& res, const defaulttype::BaseVector* v) const
    {
        tmulTranspose< Real2, FullVector<Real2>, defaulttype::BaseVector >(res, *v);
    }

    template<class Real2>
    void mul(defaulttype::BaseVector* res, const FullVector<Real2>& v) const
    {
        tmul< Real2, defaulttype::BaseVector, FullVector<Real2> >(*res, v);
    }

    template<class Real2>
    void mulTranspose(defaulttype::BaseVector* res, const FullVector<Real2>& v) const
    {
        tmulTranspose< Real2, defaulttype::BaseVector, FullVector<Real2> >(*res, v);
    }

    template<class Real2>
    void mul(defaulttype::BaseVector* res, const defaulttype::BaseVector* v) const
    {
        tmul< Real, defaulttype::BaseVector, defaulttype::BaseVector >(*res, *v);
    }

    void mulTranspose(defaulttype::BaseVector* res, const defaulttype::BaseVector* v) const
    {
        tmul< Real, defaulttype::BaseVector, defaulttype::BaseVector >(*res, *v);
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        mul(res,v);
        return res;
    }

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

    template<class Dest>
    void doCompute(Dest* dest) const
    {
        for (unsigned int xi = 0; xi < rowIndex.size(); ++xi)
        {
            Index iN = rowIndex[xi] * NL;
            Range rowRange(rowBegin[xi], rowBegin[xi+1]);
            for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index jN = colsIndex[xj] * NC;
                const Bloc& b = colsValue[xj];
                for (int bi = 0; bi < NL; ++bi)
                    for (int bj = 0; bj < NC; ++bj)
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
                for (int bi = 0; bi < NL; ++bi)
                    for (int bj = 0; bj < NC; ++bj)
                        dest->add(iN+bi, jN+bj, traits::v(b, bi, bj));
            }
        }
    }

protected:

    template<class M>
    void compute(const M& m, bool add = false)
    {
        if (m.hasRef(this))
        {
            Matrix tmp;
            tmp.resize(m.rowSize(), m.colSize());
            m.doCompute(&tmp);
            if (add)
                tmp.doCompute(this);
            else
                swap(tmp);
        }
        else
        {
            if (!add)
                resize(m.rowSize(), m.colSize());
            m.doCompute(this);
        }
    }
public:

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        if (&m == this) return;
        resize(m.rowSize(), m.colSize());
        m.doCompute(this);
    }

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator+=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        compute(m, true);
    }

    template<class TBloc2, class TVecBloc2, class TVecIndex2>
    void operator-=(const CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2>& m)
    {
        compute(MatrixExpr< MatrixNegative< CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2> > >(MatrixNegative< CompressedRowSparseMatrix<TBloc2, TVecBloc2, TVecIndex2> >(m)), true);
    }

    template<class Expr2>
    void operator=(const MatrixExpr< Expr2 >& m)
    {
        compute(m, false);
    }

    template<class Expr2>
    void operator+=(const MatrixExpr< Expr2 >& m)
    {
        compute(m, true);
    }

    template<class Expr2>
    void operator-=(const MatrixExpr< Expr2 >& m)
    {
        compute(MatrixExpr< MatrixNegative< Expr2 > >(MatrixNegative< Expr2 >(m)), true);
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

    friend std::ostream& operator << (std::ostream& out, const Matrix& v )
    {
        int nx = v.colSize();
        int ny = v.rowSize();
        out << "[";
        for (int y=0; y<ny; ++y)
        {
            out << "\n[";
            for (int x=0; x<nx; ++x)
            {
                out << " " << v.element(y,x);
            }
            out << " ]";
        }
        out << " ]";
        return out;
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
                (int *) &(this->getRowBegin()[0]),
                (int *) &(this->getColsIndex()[0]),
                (double *) &(this->getColsValue()[0])
                );
    }

    static bool check_matrix(
        int nzmax,// nb values
        int m,// number of row
        int n,// number of columns
        int * a_p,// column pointers (size n+1) or col indices (size nzmax)
        int * a_i,// row indices, size nzmax
        double * a_x// numerical values, size nzmax
    )
    {
        // check ap, size m beecause ther is at least the diagonal value wich is different of 0
        if (a_p[0]!=0)
        {
            std::cerr << "CompressedRowSparseMatrix: First value of row indices (a_p) should be 0" << std::endl;
            return false;
        }

        for (int i=1; i<=m; i++)
        {
            if (a_p[i]<=a_p[i-1])
            {
                std::cerr << "CompressedRowSparseMatrix: Row (a_p) indices are not sorted index " << i-1 << " : " << a_p[i-1] << " , " << i << " : " << a_p[i] << std::endl;
                return false;
            }
        }
        if (nzmax == -1)
        {
            nzmax = a_p[m];
        }
        else if (a_p[m]!=nzmax)
        {
            std::cerr << "CompressedRowSparseMatrix: Last value of row indices (a_p) should be " << nzmax << " and is " << a_p[m] << std::endl;
            return false;
        }


        int k=1;
        for (int i=0; i<nzmax; i++)
        {
            i++;
            for (; i<a_p[k]; i++)
            {
                if (a_i[i] <= a_i[i-1])
                {
                    std::cerr << "CompressedRowSparseMatrix: Column (a_i) indices are not sorted index " << i-1 << " : " << a_i[i-1] << " , " << i << " : " << a_p[i] << std::endl;
                    return false;
                }
                if (a_i[i]<0 || a_i[i]>=n)
                {
                    std::cerr << "CompressedRowSparseMatrix: Column (a_i) indices are not correct " << i << " : " << a_i[i] << std::endl;
                    return false;
                }
            }
            k++;
        }

        for (int i=0; i<nzmax; i++)
        {
            if (a_x[i]==0)
            {
                std::cerr << "CompressedRowSparseMatrix: Warning , matrix contains 0 , index " << i << std::endl;
                return false;
            }
        }

        if (n!=m)
        {
            std::cerr << "CompressedRowSparseMatrix: the matrix is not square" << std::endl;
            return false;
        }

        std::cerr << "Check_matrix passed successfully" << std::endl;
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
