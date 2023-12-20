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
/******************************************************************************
* Contributors:
*   - InSimo
*******************************************************************************/
#pragma once

#include <sofa/linearalgebra/config.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixGeneric.h>

#include <sofa/type/trait/is_vector.h>

namespace sofa::linearalgebra
{

/// Mechanical policy type, showing the types and flags to give to CompressedRowSparseMatrixMechanical
/// for its second template type.
class CRSMechanicalPolicy : public CRSDefaultPolicy
{
public:
    static constexpr bool CompressZeros = false; // keep old behavior
    static constexpr bool IsAlwaysSquare = true;
    static constexpr bool IsAlwaysSymmetric = true;
    static constexpr bool OrderedInsertion = false; // keep old behavior
    static constexpr bool StoreLowerTriangularBlock = true;

    static constexpr int matrixType = 1;
};

template<typename TBlock, typename TPolicy = CRSMechanicalPolicy >
class CompressedRowSparseMatrixMechanical final // final is used to allow the compiler to inline virtual methods
    : public CompressedRowSparseMatrixGeneric<TBlock, TPolicy>, public sofa::linearalgebra::BaseMatrix
{
public:
    typedef CompressedRowSparseMatrixMechanical<TBlock, TPolicy> Matrix;

    typedef CompressedRowSparseMatrixGeneric<TBlock, TPolicy> CRSMatrix;
    typedef typename CRSMatrix::Policy Policy;

    using Block     = TBlock;
    using VecBlock  = typename CRSBlockTraits<Block>::VecBlock;
    using VecIndex = typename CRSBlockTraits<Block>::VecIndex;
    using VecFlag  = typename CRSBlockTraits<Block>::VecFlag;
    using Index    = typename VecIndex::value_type;

    typedef typename CRSMatrix::Block Data;
    typedef typename CRSMatrix::Range Range;
    typedef typename CRSMatrix::traits traits;
    typedef typename CRSMatrix::Real Real;
    typedef typename CRSMatrix::Index KeyType;
    typedef typename CRSMatrix::IndexedBlock IndexedBlock;
    typedef typename CRSMatrix::VecIndexedBlock VecIndexedBlock;

    typedef Matrix Expr;
    typedef CompressedRowSparseMatrixMechanical<double> matrix_type;
    enum { category = MATRIX_SPARSE };
    enum { operand = 1 };

    enum { NL = CRSMatrix::NL };  ///< Number of rows of a block
    enum { NC = CRSMatrix::NC };  ///< Number of columns of a block

    /// Size
    Index nRow,nCol;         ///< Mathematical size of the matrix, in scalars
    static_assert(!Policy::AutoSize,
        "CompressedRowSparseMatrixMechanical cannot use AutoSize policy to make sure block-based and scalar-based sizes match");

    CompressedRowSparseMatrixMechanical()
        : CRSMatrix()
        , nRow(0), nCol(0)
    {
    }

    CompressedRowSparseMatrixMechanical(Index nbRow, Index nbCol)
        : CRSMatrix((nbRow + NL-1) / NL, (nbCol + NC-1) / NC)
        , nRow(nbRow), nCol(nbCol)
    {
    }

    static void split_row_index(Index& index, Index& modulo) { bloc_index_func<NL, Index>::split(index, modulo); }
    static void split_col_index(Index& index, Index& modulo) { bloc_index_func<NC, Index>::split(index, modulo); }

    void compress() override
    {
        CRSMatrix::compress();
    }

    void swap(Matrix& m)
    {
        Index t;
        t = nRow; nRow = m.nRow; m.nRow = t;
        t = nCol; nCol = m.nCol; m.nCol = t;
        CRSMatrix::swap(m);
    }

    /// Make sure all diagonal entries are present even if they are zero
    template< typename = typename std::enable_if< Policy::IsAlwaysSquare> >
    void fullDiagonal()
    {
        compress();
        Index ndiag = 0;
        for (Index r = 0; r < static_cast<Index>(this->rowIndex.size()); ++r)
        {
            Index i = this->rowIndex[r];
            Index b = this->rowBegin[r];
            Index e = this->rowBegin[r+1];
            Index t = b;
            while (b < e && this->colsIndex[t] != i)
            {
                if (this->colsIndex[t] < i)
                    b = t+1;
                else
                    e = t;
                t = (b+e)>>1;
            }
            if (b<e) ++ndiag;
        }
        if (ndiag == this->nBlockRow) return;

        this->oldRowIndex.swap(this->rowIndex);
        this->oldRowBegin.swap(this->rowBegin);
        this->oldColsIndex.swap(this->colsIndex);
        this->oldColsValue.swap(this->colsValue);
        this->rowIndex.resize(this->nBlockRow);
        this->rowBegin.resize(this->nBlockRow + 1);
        this->colsIndex.resize(this->oldColsIndex.size() + this->nBlockRow-ndiag);
        this->colsValue.resize(this->oldColsValue.size() + this->nBlockRow-ndiag);

        Index nv = 0;
        for (Index i = 0; i < this->nBlockRow; ++i) this->rowIndex[i] = i;
        Index j = 0;
        for (Index i = 0; i < static_cast<Index>(this->oldRowIndex.size()); ++i)
        {
            for (; j < this->oldRowIndex[i]; ++j)
            {
                this->rowBegin[j] = nv;
                this->colsIndex[nv] = j;
                traits::clear(this->colsValue[nv]);
                ++nv;
            }
            this->rowBegin[j] = nv;
            Index b = this->oldRowBegin[i];
            Index e = this->oldRowBegin[i+1];
            for (; b < e && this->oldColsIndex[b] < j; ++b)
            {
                this->colsIndex[nv] = this->oldColsIndex[b];
                this->colsValue[nv] = this->oldColsValue[b];
                ++nv;
            }
            if (b >= e || this->oldColsIndex[b] > j)
            {
                this->colsIndex[nv] = j;
                traits::clear(this->colsValue[nv]);
                ++nv;
            }
            for (; b < e; ++b)
            {
                this->colsIndex[nv] = this->oldColsIndex[b];
                this->colsValue[nv] = this->oldColsValue[b];
                ++nv;
            }
            ++j;
        }
        for (; j < this->nBlockRow; ++j)
        {
            this->rowBegin[j] = nv;
            this->colsIndex[nv] = j;
            traits::clear(this->colsValue[nv]);
            ++nv;
        }
        this->rowBegin[j] = nv;
    }

    ///< Mathematical size of the matrix
    Index rowSize() const override
    {
        return nRow;
    }

    ///< Mathematical size of the matrix
    Index colSize() const override
    {
        return nCol;
    }

    /// This override classic resizeBlock to fill nRow and nCol values.
    void resizeBlock(Index nbBRow, Index nbBCol) override
    {
        CRSMatrix::resizeBlock(nbBRow, nbBCol);
        nRow = NL * nbBRow;
        nCol = NC * nbBCol;
    }

    void resize(Index nbRow, Index nbCol) override
    {
        this->resizeBlock((nbRow + NL-1) / NL, (nbCol + NC-1) / NC);
        nRow = nbRow;
        nCol = nbCol;
    }

    void extend(Index nbRow, Index nbCol)
    {
        nRow = nbRow;
        nCol = nbCol;
        this->nBlockRow = (nbRow + NL-1) / NL;
        this->nBlockCol = (nbCol + NC-1) / NC;
    }

    /**
    * \brief get scalar element i, j of matrix
    **/
    SReal element(Index i, Index j) const override
    {
        if constexpr (Policy::AutoCompress)
        {
            const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        }
        
        if constexpr (!Policy::StoreLowerTriangularBlock) if ((i / NL) > (j / NC))
        {
            std::swap(i, j);
        }

        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        return traits::v(this->block(i, j), bi, bj);
    }

    /**
    * \brief set scalar element i, j of matrix
    **/
    void set(Index i, Index j, double v) override
    {
        if constexpr (!Policy::StoreLowerTriangularBlock)
        {
            if ((i / NL) > (j / NC)) 
                return;
        }
        
        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        traits::vset(*this->wblock(i, j, true), bi, bj, static_cast<Real>(v) );
    }

    /**
    * \brief add scalar v at element i, j of matrix
    **/
    void add(Index i, Index j, double v) override
    {
        if constexpr (!Policy::StoreLowerTriangularBlock)
        {
            if ((i / NL) > (j / NC)) 
                return;
        }

        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        traits::vadd(*this->wblock(i,j,true), bi, bj, static_cast<Real>(v) );
    }

    /**
    * \brief set scalar element i, j of matrix when rowId and colId are known
    **/
    void set(Index i, Index j, int& rowId, int& colId, double v)
    {
        if constexpr (!Policy::StoreLowerTriangularBlock) if ((i / NL) > (j / NC)) return;

        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        traits::vset(*this->wblock(i,j,rowId,colId,true), bi, bj, static_cast<Real>(v) );
    }

    /**
    * \brief add scalar v at element i, j when rowId and colId are known
    **/
    template <typename T = Block, typename std::enable_if_t<!std::is_same_v<T, double> && !std::is_same_v<T, float>, int > = 0 >
    void add(Index i, Index j, int& rowId, int& colId, double v)
    {
        if constexpr (!Policy::StoreLowerTriangularBlock)
        {
            if ((i / NL) > (j / NC)) 
                return;
        }

        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        traits::vadd(*this->wblock(i,j,rowId,colId,true), bi, bj, static_cast<Real>(v) );
    }

    /**
    * \brief clear scalar at element i, j of matrix
    **/
    void clear(Index i, Index j) override
    {
        if constexpr (Policy::AutoCompress) this->compress();

        Index bi=0, bj=0; split_row_index(i, bi); split_col_index(j, bj);
        Block* b = this->wblock(i,j,false);

        if (b) 
            traits::vset(*b, bi, bj, 0);
    }

    void add(Index i, Index j, const type::Mat3x3d& _M) override
    {
        BaseMatrix::add(i, j, _M);
    }

    void add(Index i, Index j, const type::Mat3x3f& _M) override
    {
        BaseMatrix::add(i, j, _M);
    }

    /**
    * \brief Clear row scalar method. Clear all col of this line.
    * @param i : Line index considering size of matrix in scalar.
    * \warning If you want to clear all value of a block, it is better to call clearRowBlock
    **/
    void clearRow(Index i) override
    {
        if constexpr (Policy::AutoCompress) this->compress(); /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.

        Index bi=0; split_row_index(i, bi);
        Index rowId = Index(i * this->rowIndex.size() / this->nBlockRow);
        if (this->sortedFind(this->rowIndex, i, rowId))
        {
            Range rowRange(this->rowBegin[rowId], this->rowBegin[rowId+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Block& b = this->colsValue[xj];
                for (Index bj = 0; bj < NC; ++bj)
                    traits::vset(b, bi, bj, 0);
            }
        }
    }

    /**
    * \brief Clear col scalar method. Clear this col in all row of matrix.
    * @param j : Col index considering size of matrix in scalar.
    * \warning If you want to clear all value of a block, it is better to call clearColBlock
    **/
    void clearCol(Index j) override
    {
        /// If AutoCompress policy is activated, we neeed to be sure not missing btemp registered value.
        if constexpr (Policy::AutoCompress) this->compress();

        Index bj = 0; split_col_index(j, bj);
        for (Index i = 0; i < this->nBlockRow; ++i)
        {
            Block* b = this->wblock(i,j,false);
            if (b)
            {
                for (Index bi = 0; bi < NL; ++bi)
                    traits::vset(*b, bi, bj, 0);
            }
        }
    }

    /**
    * \brief Clear both row i and column i in a square matrix
    * @param i : Row and Col index considering size of matrix in scalar.
    **/
    void clearRowCol(Index i) override
    {
        if constexpr (!Policy::IsAlwaysSquare || !Policy::StoreLowerTriangularBlock)
        {
            clearRow(i);
            clearCol(i);
        }
        else
        {
            /// If AutoCompress policy is activated, we need to be sure that we are not missing btemp registered value.
            if constexpr (Policy::AutoCompress) this->compress();

            Index bi=0; split_row_index(i, bi);
            Index rowId = Index(i * this->rowIndex.size() / this->nBlockRow);
            if (this->sortedFind(this->rowIndex, i, rowId))
            {
                Range rowRange(this->rowBegin[rowId], this->rowBegin[rowId+1]);
                for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
                {
                    Block* b = &this->colsValue[xj];
                    // first clear (i,j)
                    for (Index bj = 0; bj < NC; ++bj)
                        traits::vset(*b, bi, bj, 0);

                    // then clear (j,i) 
                    Index j = this->colsIndex[xj];
                    
                    if (j != i)
                    {
                        Range jrowRange(this->rowBegin[j], this->rowBegin[j + 1]);
                        Index colId = 0;

                        // look for column i
                        if (this->sortedFind(this->colsIndex, jrowRange, i, colId))
                        {
                            b = &this->colsValue[colId];
                        }
                    }

                    for (Index bj = 0; bj < NL; ++bj)
                        traits::vset(*b, bj, bi, 0);
             
                }
            }
        }
    }

    /**
    * \brief Completely clear the matrix
    **/
    void clear() override /// Need implement clear to override BaseMatrix one.
    {
        CRSMatrix::clear();
    }

/// @name BlockMatrixWriter operators
/// @{
    /// Override CRSMatrix add method to avoid mis-understanding by compiler with other add method overriding BaseMatrix.
    template <typename T = Block, typename std::enable_if_t<!std::is_same_v<T, double> && !std::is_same_v<T, float>, int > = 0 >
    void add(unsigned int bi, unsigned int bj, const Block& b)
    {
        CRSMatrix::add(bi, bj, b);
    }
/// @}

/// @name Get information about the content and structure of this matrix (diagonal, band, sparse, full, block size, ...)
/// @{

    /// @return type of elements stored in this matrix
    virtual ElementType getElementType() const override { return traits::getElementType(); }

    /// @return size of elements stored in this matrix
    virtual std::size_t getElementSize() const override { return sizeof(Real); }

    /// @return the category of this matrix
    virtual MatrixCategory getCategory() const override { return MATRIX_SPARSE; }

    /// @return the number of rows in each block, or 1 of there are no fixed block size
    virtual Index getBlockRows() const override { return NL; }

    /// @return the number of columns in each block, or 1 of there are no fixed block size
    virtual Index getBlockCols() const override { return NC; }

    /// @return the number of rows of blocks
    virtual Index bRowSize() const override{ return this->rowBSize(); }

    /// @return the number of columns of blocks
    virtual Index bColSize() const override { return this->colBSize(); }

    /// @return the width of the band on each side of the diagonal (only for band matrices)
    virtual Index getBandWidth() const override { return NC-1; }

/// @}

/// @name Filtering-out part of a matrix
/// @{

    typedef bool filter_fn     (Index   i  , Index   j  , Block& val, const Real   ref  );
    static bool nonzeros(Index /*i*/, Index /*j*/, Block& val, const Real /*ref*/) { return (!traits::empty(val)); }
    static bool nonsmall(Index /*i*/, Index /*j*/, Block& val, const Real   ref  )
    {
        for (Index bi = 0; bi < NL; ++bi)
            for (Index bj = 0; bj < NC; ++bj)
                if (type::rabs(traits::v(val, bi, bj)) >= ref) return true;
        return false;
    }
    static bool upper(Index   i  , Index   j  , Block& val, const Real /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (Index bi = 1; bi < NL; ++bi)
                for (Index bj = 0; bj < bi; ++bj)
                    traits::vset(val, bi, bj, 0);
        }
        return i*NL <= j*NC;
    }
    static bool lower(Index   i  , Index   j  , Block& val, const Real /*ref*/)
    {
        if (NL>1 && i*NL == j*NC)
        {
            for (Index bi = 0; bi < NL-1; ++bi)
                for (Index bj = bi+1; bj < NC; ++bj)
                    traits::vset(val, bi, bj, 0);
        }
        return i*NL >= j*NC;
    }
    static bool upper_nonzeros(Index   i  , Index   j  , Block& val, const Real   ref  ) { return upper(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool lower_nonzeros(Index   i  , Index   j  , Block& val, const Real   ref  ) { return lower(i,j,val,ref) && nonzeros(i,j,val,ref); }
    static bool upper_nonsmall(Index   i  , Index   j  , Block& val, const Real   ref  ) { return upper(i,j,val,ref) && nonsmall(i,j,val,ref); }
    static bool lower_nonsmall(Index   i  , Index   j  , Block& val, const Real   ref  ) { return lower(i,j,val,ref) && nonsmall(i,j,val,ref); }

    template<class TMatrix>
    void filterValues(TMatrix& M, filter_fn* filter = &nonzeros, const Real ref = Real(), bool keepEmptyRows=false)
    {
        M.compress();
        this->nBlockRow = M.rowBSize();
        this->nBlockCol = M.colBSize();
        this->nRow = M.rowSize();
        this->nCol = M.colSize();
        this->rowIndex.clear();
        this->rowBegin.clear();
        this->colsIndex.clear();
        this->colsValue.clear();
        this->skipCompressZero = true;
        this->btemp.clear();
        this->rowIndex.reserve(M.rowIndex.size());
        this->rowBegin.reserve(M.rowBegin.size());
        this->colsIndex.reserve(M.colsIndex.size());
        this->colsValue.reserve(M.colsValue.size());

        Index vid = 0;
        for (Index rowId = 0; rowId < static_cast<Index>(M.rowIndex.size()); ++rowId)
        {
            Index i = M.rowIndex[rowId];
            this->rowIndex.push_back(i);
            this->rowBegin.push_back(vid);
            Range rowRange(M.rowBegin[rowId], M.rowBegin[rowId+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index j = M.colsIndex[xj];
                Block b = M.colsValue[xj];
                if ((*filter)(i,j,b,ref))
                {
                    this->colsIndex.push_back(j);
                    this->colsValue.push_back(b);
                    ++vid;
                }
            }
            if (!keepEmptyRows && this->rowBegin.back() == vid) // row was empty
            {
                this->rowIndex.pop_back();
                this->rowBegin.pop_back();
            }
        }
        this->rowBegin.push_back(vid); // end of last row
    }

    template <class TMatrix>
    void copyNonZeros(TMatrix& M, bool keepEmptyRows=false)
    {
        filterValues(M, nonzeros, Real(), keepEmptyRows);
    }

    template <class TMatrix>
    void copyNonSmall(TMatrix& M, const Real ref, bool keepEmptyRows=false)
    {
        filterValues(M, nonsmall, ref, keepEmptyRows);
    }

    void copyUpper(Matrix& M, bool keepEmptyRows=false)
    {
        filterValues(M, upper, Real(), keepEmptyRows);
    }

    void copyLower(Matrix& M, bool keepEmptyRows=false)
    {
        filterValues(M, lower, Real(), keepEmptyRows);
    }

    template <class TMatrix>
    void copyUpperNonZeros(TMatrix& M, bool keepEmptyRows=false)
    {
        filterValues(M, upper_nonzeros, Real(), keepEmptyRows);
    }

    template <class TMatrix>
    void copyLowerNonZeros(TMatrix& M, bool keepEmptyRows=false)
    {
        filterValues(M, lower_nonzeros, Real(), keepEmptyRows);
    }

    void copyUpperNonSmall(Matrix& M, const Real ref, bool keepEmptyRows=false)
    {
        filterValues(M, upper_nonsmall, ref, keepEmptyRows);
    }

    void copyLowerNonSmall(Matrix& M, const Real ref, bool keepEmptyRows=false)
    {
        filterValues(M, lower_nonsmall, ref, keepEmptyRows);
    }

/// @}

/// @name Virtual iterator classes and methods
/// @{

protected:
    virtual void bAccessorDelete(const InternalBlockAccessor* /*b*/) const override {}
    virtual void bAccessorCopy(InternalBlockAccessor* /*b*/) const override {}
    virtual SReal bAccessorElement(const InternalBlockAccessor* b, Index i, Index j) const override
    {
        //return element(b->row * getBlockRows() + i, b->col * getBlockCols() + j);
        Index index = b->data;
        const Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        return static_cast<SReal>(traits::v(data, i, j));
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, Index i, Index j, double v) override
    {
        //set(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
        Index index = b->data;
        Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        traits::vset(data, i, j, static_cast<Real>(v) );
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, Index i, Index j, double v) override
    {
        //add(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
        Index index = b->data;
        Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        traits::vadd(data, i, j, static_cast<Real>(v) );
    }

    template<class T>
    const T* bAccessorElementsCSRImpl(const InternalBlockAccessor* b, T* buffer) const
    {
        Index index = b->data;
        const Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                buffer[l*NC+c] = static_cast<T>(traits::v(data, l, c));
        return buffer;
    }
    virtual const float* bAccessorElements(const InternalBlockAccessor* b, float* buffer) const override
    {
        return bAccessorElementsCSRImpl<float>(b, buffer);
    }
    virtual const double* bAccessorElements(const InternalBlockAccessor* b, double* buffer) const override
    {
        return bAccessorElementsCSRImpl<double>(b, buffer);
    }
    virtual const int* bAccessorElements(const InternalBlockAccessor* b, int* buffer) const override
    {
        return bAccessorElementsCSRImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorSetCSRImpl(InternalBlockAccessor* b, const T* buffer)
    {
        Index index = b->data;
        Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                traits::vset(data, l, c, static_cast<Real>(buffer[l*NC+c]) );
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const float* buffer) override
    {
        bAccessorSetCSRImpl<float>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const double* buffer) override
    {
        bAccessorSetCSRImpl<double>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const int* buffer) override
    {
        bAccessorSetCSRImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorAddCSRImpl(InternalBlockAccessor* b, const T* buffer)
    {
        Index index = b->data;
        Block& data = (index >= 0) ? this->colsValue[index] : this->btemp[-index-1].value;
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                traits::vadd(data, l, c,static_cast<Real>(buffer[l*NC+c]) );
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const float* buffer) override
    {
        bAccessorAddCSRImpl<float>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const double* buffer) override
    {
        bAccessorAddCSRImpl<double>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const int* buffer) override
    {
        bAccessorAddCSRImpl<int>(b, buffer);
    }

public:

    /// Get read access to a block
    virtual BlockConstAccessor blockGet(Index i, Index j) const
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !

        Index rowId = Index(i * this->rowIndex.size() / this->nBlockRow);
        if (this->sortedFind(this->rowIndex, i, rowId))
        {
            Range rowRange(this->rowBegin[rowId], this->rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / this->nBlockCol;
            if (this->sortedFind(this->colsIndex, rowRange, j, colId))
            {
                return createBlockConstAccessor(i, j, colId);
            }
        }
        return createBlockConstAccessor(-1-i, -1-j, static_cast<Index>(0));
    }

    /// Get write access to a block
    virtual BlockAccessor blockGetW(Index i, Index j)
    {
        if constexpr (Policy::AutoCompress) compress();

        Index rowId = Index(i * this->rowIndex.size() / this->nBlockRow);
        if (this->sortedFind(this->rowIndex, i, rowId))
        {
            Range rowRange(this->rowBegin[rowId], this->rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / this->nBlockCol;
            if (this->sortedFind(this->colsIndex, rowRange, j, colId))
            {
                return createBlockAccessor(i, j, colId);
            }
        }
        return createBlockAccessor(-1-i, -1-j, static_cast<Index>(0));
    }

    /// Get write access to a block, possibly creating it
    virtual BlockAccessor blockCreate(Index i, Index j)
    {
        Index rowId = Index(i * this->rowIndex.size() / this->nBlockRow);
        if (this->sortedFind(this->rowIndex, i, rowId))
        {
            Range rowRange(this->rowBegin[rowId], this->rowBegin[rowId+1]);
            Index colId = rowRange.begin() + j * rowRange.size() / this->nBlockCol;
            if (this->sortedFind(this->colsIndex, rowRange, j, colId))
            {
                return createBlockAccessor(i, j, colId);
            }
        }
        if (this->btemp.empty() || this->btemp.back().l != i || this->btemp.back().c != j)
        {
            this->btemp.push_back(IndexedBlock(i,j));
            traits::clear(this->btemp.back().value);
        }
        return createBlockAccessor(i, j, -static_cast<Index>(this->btemp.size()));
    }

protected:
    virtual void itCopyColBlock(InternalColBlockIterator* /*it*/) const override {}
    virtual void itDeleteColBlock(const InternalColBlockIterator* /*it*/) const override {}
    virtual void itAccessColBlock(InternalColBlockIterator* it, BlockConstAccessor* b) const override
    {
        Index index = it->data;
        setMatrix(b);
        getInternal(b)->row = it->row;
        getInternal(b)->data = index;
        getInternal(b)->col = this->colsIndex[index];
    }
    virtual void itIncColBlock(InternalColBlockIterator* it) const override
    {
        Index index = it->data;
        ++index;
        it->data = index;
    }
    virtual void itDecColBlock(InternalColBlockIterator* it) const override
    {
        Index index = it->data;
        --index;
        it->data = index;
    }
    virtual bool itEqColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const override
    {
        Index index = it->data;
        Index index2 = it2->data;
        return index == index2;
    }
    virtual bool itLessColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const override
    {
        Index index = it->data;
        Index index2 = it2->data;
        return index < index2;
    }

public:
    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(Index ib) const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        Index rowId = Index(ib * this->rowIndex.size() / this->nBlockRow);
        Index index = 0;
        if (this->sortedFind(this->rowIndex, ib, rowId))
        {
            index = this->rowBegin[rowId];
        }
        return createColBlockConstIterator(ib, index);
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(Index ib) const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        Index rowId = Index(ib * this->rowIndex.size() / this->nBlockRow);
        Index index2 = 0;
        if (this->sortedFind(this->rowIndex, ib, rowId))
        {
            index2 = this->rowBegin[rowId+1];
        }
        return createColBlockConstIterator(ib, index2);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(Index ib) const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        Index rowId = Index(ib * this->rowIndex.size() / this->nBlockRow);
        Index index = 0, index2 = 0;
        if (this->sortedFind(this->rowIndex, ib, rowId))
        {
            index = this->rowBegin[rowId];
            index2 = this->rowBegin[rowId+1];
        }
        return std::make_pair(createColBlockConstIterator(ib, index ),
                createColBlockConstIterator(ib, index2));
    }


protected:
    virtual void itCopyRowBlock(InternalRowBlockIterator* /*it*/) const override {}
    virtual void itDeleteRowBlock(const InternalRowBlockIterator* /*it*/) const override {}
    virtual Index itAccessRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        return this->rowIndex[rowId];
    }
    virtual ColBlockConstIterator itBeginRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        Index row = this->rowIndex[rowId];
        Index index = this->rowBegin[rowId];
        return createColBlockConstIterator(row, index);
    }
    virtual ColBlockConstIterator itEndRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        Index row = this->rowIndex[rowId];
        Index index2 = this->rowBegin[rowId+1];
        return createColBlockConstIterator(row, index2);
    }
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> itRangeRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        Index row = this->rowIndex[rowId];
        Index index = this->rowBegin[rowId];
        Index index2 = this->rowBegin[rowId+1];
        return std::make_pair(createColBlockConstIterator(row, index ),
                createColBlockConstIterator(row, index2));
    }

    virtual void itIncRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        ++rowId;
        it->data[0] = rowId;
    }
    virtual void itDecRowBlock(InternalRowBlockIterator* it) const override
    {
        Index rowId = it->data[0];
        --rowId;
        it->data[0] = rowId;
    }
    virtual bool itEqRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const override
    {
        Index rowId = it->data[0];
        Index rowId2 = it2->data[0];
        return rowId == rowId2;
    }
    virtual bool itLessRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const override
    {
        Index rowId = it->data[0];
        Index rowId2 = it2->data[0];
        return rowId < rowId2;
    }

public:
    /// Get the iterator corresponding to the beginning of the rows of blocks
    virtual RowBlockConstIterator bRowsBegin() const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        return createRowBlockConstIterator(0, 0);
    }

    /// Get the iterator corresponding to the end of the rows of blocks
    virtual RowBlockConstIterator bRowsEnd() const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        return createRowBlockConstIterator(Index(this->rowIndex.size()), 0);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<RowBlockConstIterator, RowBlockConstIterator> bRowsRange() const override
    {
        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        return std::make_pair(createRowBlockConstIterator(0, 0),
                createRowBlockConstIterator(Index(this->rowIndex.size()), 0));
    }

/// @}

protected:

/// @name setter/getter & product methods on template vector types
/// @{

    template<class Vec> static Real vget(const Vec& vec, Index i, Index j, Index k) { return vget( vec, i*j+k ); }
    template<class Vec> static Real vget(const type::vector<Vec>&vec, Index i, Index /*j*/, Index k) { return vec[i][k]; }

                          static Real  vget(const BaseVector& vec, Index i) { return static_cast<Real>(vec.element(i)); }
    template<class Real2> static Real2 vget(const FullVector<Real2>& vec, Index i) { return vec[i]; }


    template<class Vec> static void vset(Vec& vec, Index i, Index j, Index k, Real v) { vset( vec, i*j+k, v ); }
    template<class Vec> static void vset(type::vector<Vec>&vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] = v; }

                          static void vset(BaseVector& vec, Index i, Real v) { vec.set(i, v); }
    template<class Real2> static void vset(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] = v; }


    template<class Vec> static void vadd(Vec& vec, Index i, Index j, Index k, Real v) { vadd( vec, i*j+k, v ); }
    template<class Vec> static void vadd(type::vector<Vec>&vec, Index i, Index /*j*/, Index k, Real v) { vec[i][k] += v; }

                          static void vadd(BaseVector& vec, Index i, Real v) { vec.add(i, v); }
    template<class Real2> static void vadd(FullVector<Real2>& vec, Index i, Real2 v) { vec[i] += v; }

    template<class Vec> static void vresize(Vec& vec, Index /*blockSize*/, Index totalSize) { vec.resize( totalSize ); }
    template<class Vec> static void vresize(type::vector<Vec>&vec, Index blockSize, Index /*totalSize*/) { vec.resize( blockSize ); }


    /** Product of the matrix with a templated vector res = this * vec*/
    //template<class Real2, class V1, class V2>
    //void tmul(V1& res, const V2& vec) const
    //{
    //    assert( vec.size() % bColSize() == 0 ); // vec.size() must be a multiple of block size.

    //    if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !

    //    vresize( res, this->rowBSize(), rowSize() );

    //    for (Index xi = 0; xi < static_cast<Index>(this->rowIndex.size()); ++xi)  // for each non-empty block row
    //    {
    //        Range rowRange(this->rowBegin[xi], this->rowBegin[xi+1]);
    //        for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
    //        {
    //            sofa::type::Vec<NL,Real2> vi;
    //            const Block& b = this->colsValue[xj];
    //            Index rowIndex = this->rowIndex[xi] * NL;
    //            Index colIndex = this->colsIndex[xj] * NC;
    //            std::copy(vec.begin() + colIndex, vec.begin() + colIndex + NC, vi.begin());
    //            for (Index bi = 0; bi < NL; ++bi)
    //                for (Index bj = 0; bj < NC; ++bj)
    //                    res[rowIndex + bi] += traits::v(b, bi, bj) * vi[bj];

    //            if constexpr (!Policy::StoreLowerTriangularBlock)
    //            {
    //                if (colIndex != rowIndex)
    //                {
    //                    sofa::type::Vec<NL,Real2> vj;
    //                    std::copy(vec.begin() + rowIndex, vec.begin() + rowIndex + NL, vj.begin());
    //                    for (Index bi = 0; bi < NL; ++bi)
    //                        for (Index bj = 0; bj < NC; ++bj)
    //                            res[colIndex + bi] += traits::v(b, bj, bi) * vj[bj];
    //                }
    //            }
    //        }
    //    }
    //}

    /** Product of the matrix with a templated vector res = this * vec*/
    template<class Real2, class V1, class V2>
    void tmul(V1& res, const V2& vec) const
    {
        assert(vec.size() % bColSize() == 0); // vec.size() must be a multiple of block size.

        ((Matrix*)this)->compress();
        vresize(res, this->rowBSize(), this->rowSize());
        for (Index xi = 0; xi < (Index)this->rowIndex.size(); ++xi)  // for each non-empty block row
        {
            type::Vec<NL, Real2> r;  // local block-sized vector to accumulate the product of the block row  with the large vector

            // multiply the non-null blocks with the corresponding chunks of the large vector
            Range rowRange(this->rowBegin[xi], this->rowBegin[xi + 1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                // transfer a chunk of large vector to a local block-sized vector
                type::Vec<NC, Real2> v;
                //Index jN = colsIndex[xj] * NC;    // scalar column index
                for (Index bj = 0; bj < NC; ++bj)
                    v[bj] = vget(vec, this->colsIndex[xj], NC, bj);

                // multiply the block with the local vector
                const Block& b = this->colsValue[xj];    // non-null block has block-indices (rowIndex[xi],colsIndex[xj]) and value colsValue[xj]
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        r[bi] += traits::v(b, bi, bj) * v[bj];
            }

            // transfer the local result  to the large result vector
            //Index iN = rowIndex[xi] * NL;                      // scalar row index
            for (Index bi = 0; bi < NL; ++bi)
                vset(res, this->rowIndex[xi], NL, bi, r[bi]);
        }
    }


    /** Product of the matrix with a templated vector res += this * vec*/
    template<class Real2, class V1, class V2>
    void taddMul(V1& res, const V2& vec) const
    {
        assert( vec.size()%bColSize() == 0 ); // vec.size() must be a multiple of block size.

        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        vresize( res, this->rowBSize(), rowSize() );

        for (Index xi = 0; xi < static_cast<Index>(this->rowIndex.size()); ++xi)
        {
            Range rowRange(this->rowBegin[xi], this->rowBegin[xi+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                sofa::type::Vec<NL,Real2> vi;
                const Block& b = this->colsValue[xj];
                Index rowIndex = this->rowIndex[xi] * NL;
                Index colIndex = this->colsIndex[xj] * NC;
                std::copy(vec.begin() + colIndex, vec.begin() + colIndex + NC, vi.begin());
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        res[rowIndex + bi] += traits::v(b, bi, bj) * vi[bj];

                if constexpr (!Policy::StoreLowerTriangularBlock)
                {
                    if (colIndex != rowIndex)
                    {
                        sofa::type::Vec<NL,Real2> vj;
                        std::copy(vec.begin() + rowIndex, vec.begin() + rowIndex + NL, vj.begin());
                        for (Index bi = 0; bi < NL; ++bi)
                            for (Index bj = 0; bj < NC; ++bj)
                                res[colIndex + bi] += traits::v(b, bj, bi) * vj[bj];
                    }
                }
            }
        }
    }


    /** Product of the matrix with a templated vector that have the size of the block res += this * [vec,...,vec]^T */
    template<class Real2, class V1, class V2>
    void taddMul_by_line(V1& res, const V2& vec) const
    {
        assert( vec.size() == NC ); // vec.size() must have the block size.

        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        vresize( res, this->rowBSize(), rowSize() );

        for (Index xi = 0; xi < static_cast<Index>(this->rowIndex.size()); ++xi)
        {
            Range rowRange(this->rowBegin[xi], this->rowBegin[xi+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                const Block& b = this->colsValue[xj];
                Index rowIndex = this->rowIndex[xi] * NL;
                Index colIndex = this->colsIndex[xj] * NC;
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        res[rowIndex + bi] += traits::v(b, bi, bj) * vec[bj];

                if constexpr (!Policy::StoreLowerTriangularBlock)
                {
                    if (colIndex != rowIndex)
                    {
                        for (Index bi = 0; bi < NL; ++bi)
                            for (Index bj = 0; bj < NC; ++bj)
                                res[colIndex + bi] += traits::v(b, bj, bi) * vec[bj];
                    }
                }
            }
        }
    }

    /** Product of the transpose with a templated vector and add it to res   res += this^T * vec */
    template<class Real2, class V1, class V2>
    void taddMulTranspose(V1& res, const V2& vec) const
    {
        assert( vec.size()%bRowSize() == 0 ); // vec.size() must be a multiple of block size.

        if constexpr (Policy::AutoCompress) const_cast<Matrix*>(this)->compress(); /// \warning this violates the const-ness of the method !
        vresize( res, this->colBSize(), colSize() );

        if constexpr (Policy::IsAlwaysSymetric) /// In symetric case this^T = this
        {
            taddMul(res, vec);
            return;
        }

        for (Index xi = 0; xi < this->rowIndex.size(); ++xi) // for each non-empty block row (i.e. column of the transpose)
        {
            // copy the corresponding chunk of the input to a local vector
            type::Vec<NL,Real2> v;
            //Index iN = rowIndex[xi] * NL;    // index of the row in the vector
            for (Index bi = 0; bi < NL; ++bi)
                v[bi] = vget(vec, this->rowIndex[xi], NL, bi);

            // accumulate the product of the column with the local vector
            Range rowRange(this->rowBegin[xi], this->rowBegin[xi+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj) // for each non-empty block in the row
            {
                const Block& b = this->colsValue[xj]; // non-empty block

                type::Vec<NC,Real2> r;  // local vector to store the product
                //Index jN = colsIndex[xj] * NC;

                // columnwise block-vector product
                for (Index bj = 0; bj < NC; ++bj)
                    r[bj] = traits::v(b, 0, bj) * v[0];
                for (Index bi = 1; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        r[bj] += traits::v(b, bi, bj) * v[bi];

                // accumulate the product to the result
                for (Index bj = 0; bj < NC; ++bj)
                    vadd(res, this->colsIndex[xj], NC, bj, r[bj]);
            }
        }
    }
    /// @}


public:

    /// @name Matrix operators
    /// @{
    
    /** @returns this + m
      @warning The block must be the same (same type and same size)
      @warning The matrices must have the same mathematical size
      @warning matrices this and m must be compressed
    */
    CompressedRowSparseMatrixMechanical<TBlock, TPolicy> operator+(const CompressedRowSparseMatrixMechanical<TBlock, TPolicy>& m) const
    {
        CompressedRowSparseMatrixMechanical<TBlock, TPolicy> res = *this;
        res += m;
        return res;
    }

    using CompressedRowSparseMatrixGeneric<TBlock, TPolicy>::mul; // CRS x CRS mul version

    /// equal result = this * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template< typename V1, typename V2, std::enable_if_t<sofa::type::trait::is_vector<V1>::value && sofa::type::trait::is_vector<V2>::value, int> = 0 >
    void mul( V2& result, const V1& v ) const
    {
        this-> template tmul< Real, V2, V1 >(result, v);
    }


    /// equal result += this^T * v
    /// @warning The block sizes must be compatible ie v.size() must be a multiple of block size.
    template< typename V1, typename V2 >
    void addMultTranspose( V1& result, const V2& v ) const
    {
        this-> template taddMulTranspose< Real, V1, V2 >(result, v);
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
    /// v has the size of one block
    template< typename V, typename Real2 >
    void addMul_by_line( V& res, const type::Vec<NC,Real2>& v ) const
    {
        this-> template taddMul_by_line< Real2,V,type::Vec<NC,Real2> >( res, v );
    }
    template< typename Real, typename V, typename V2 >
    void addMul_by_line( V& res, const V2& v ) const
    {
        this-> template taddMul_by_line< Real,V,V2 >( res, v );
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
    /// different block types possible
    /// @todo how to optimize when same block types
    template<class Dest>
    void addTo(Dest* dest) const
    {
        for (Index xi = 0; xi < static_cast<Index>(this->rowIndex.size()); ++xi)
        {
            Index iN = this->rowIndex[xi] * NL;
            Range rowRange(this->rowBegin[xi], this->rowBegin[xi+1]);
            for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
            {
                Index jN = this->colsIndex[xj] * NC;
                const Block& b = this->colsValue[xj];
                for (Index bi = 0; bi < NL; ++bi)
                    for (Index bj = 0; bj < NC; ++bj)
                        dest->add(iN+bi, jN+bj, traits::v(b, bi, bj));
            }
        }
        if (!this->btemp.empty())
        {
            for (typename VecIndexedBlock::const_iterator it = this->btemp.begin(), itend = this->btemp.end(); it != itend; ++it)
            {
                Index iN = it->l * NL;
                Index jN = it->c * NC;
                const Block& b = it->value;
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

    template<class TBlock2, class TPolicy2>
    void operator=(const CompressedRowSparseMatrixMechanical<TBlock2, TPolicy2>& m)
    {
        if (&m == this) return;
        resize(m.rowSize(), m.colSize());
        m.addTo(this);
    }

    template<class TBlock2, class TPolicy2>
    void operator+=(const CompressedRowSparseMatrixMechanical<TBlock2, TPolicy2>& m)
    {
        addEqual(m);
    }

    template<class TBlock2, class TPolicy2>
    void operator-=(const CompressedRowSparseMatrixMechanical<TBlock2, TPolicy2>& m)
    {
        equal(MatrixExpr { MatrixNegative< CompressedRowSparseMatrixMechanical<TBlock2, TPolicy2> >(m) }, true);
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
        addEqual(MatrixExpr{ MatrixNegative< Expr2 >(m) } );
    }

    MatrixExpr< MatrixTranspose< Matrix > > t() const
    {
        return MatrixExpr{ MatrixTranspose< Matrix >{*this} };
    }


    MatrixExpr< MatrixNegative< Matrix > > operator-() const
    {
        return MatrixExpr{ MatrixNegative< Matrix >(*this) };
    }

    MatrixExpr< MatrixScale< Matrix, double > > operator*(const double& r) const
    {
        return MatrixExpr{ MatrixScale< Matrix, double >(*this, r) };
    }


    static const char* Name()
    {
        // Note: to preserve backward compatibility, CompressedRowSparseMatrixMechanical keeps the same
        // name as CompressedRowSparseMatrixGeneric. We could change it later but it requires either being
        // sure all old code/scenes are updated, or add an alias mechanism in template names.
        return CRSMatrix::Name();
        // static std::string name = std::string("CompressedRowSparseMatrixMechanical") + std::string(traits::Name());
        // return name.c_str();
    }
};


template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3d >::add(Index row, Index col, const type::Mat3x3d& _M);
template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3d >::add(Index row, Index col, const type::Mat3x3f& _M);
template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3f >::add(Index row, Index col, const type::Mat3x3d& _M);
template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3f >::add(Index row, Index col, const type::Mat3x3f& _M);

template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<double>::filterValues<CompressedRowSparseMatrixMechanical<type::Mat3x3d > >(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<double>::filterValues<CompressedRowSparseMatrixMechanical<type::Mat3x3f > >(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<float>::filterValues<CompressedRowSparseMatrixMechanical<type::Mat3x3f > >(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, float> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows);
template<> template<> void SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<float>::filterValues<CompressedRowSparseMatrixMechanical<type::Mat3x3d > >(CompressedRowSparseMatrixMechanical<type::Mat<3, 3, double> >& M, filter_fn* filter, const Real ref, bool keepEmptyRows);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_COMPRESSEDROWSPARSEMATRIXMECHANICAL_CPP) 
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<float>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat1x1f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat2x2f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat4x4f>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat<6, 6, float> >;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat<8, 8, float> >;

extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<double>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat1x1d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat2x2d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat3x3d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat4x4d>;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat<6, 6, double> >;
extern template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixMechanical<type::Mat<8, 8, double> >;

#endif

} // namespace sofa::linearalgebra
