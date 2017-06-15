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
#ifndef SOFA_DEFAULTTYPE_BASEMATRIX_H
#define SOFA_DEFAULTTYPE_BASEMATRIX_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/defaulttype.h>
#include <sofa/defaulttype/BaseVector.h>
#include <utility> // for std::pair
#include <cstddef> // for NULL and std::size_t
#include <iostream>
#include <vector>
#include <cassert>
#include <limits.h>

namespace sofa
{

namespace defaulttype
{

/// Generic matrix API, allowing to fill and use a matrix independently of the linear algebra library in use.
///
/// Note that accessing values using this class is rather slow and should only be used in codes where the
/// provided genericity is necessary.
class SOFA_DEFAULTTYPE_API BaseMatrix
{
public:
    typedef BaseVector::Index Index;

    BaseMatrix();
    virtual ~BaseMatrix();

    /// Number of rows
    virtual Index rowSize(void) const = 0;
    /// Number of columns
    virtual Index colSize(void) const = 0;
    /// Number of rows (Eigen-compatible API)
    inline Index rows(void) const { return rowSize(); }
    /// Number of columns (Eigen-compatible API)
    inline Index cols(void) const { return colSize(); }
    /// Read the value of the element at row i, column j (using 0-based indices)
    virtual SReal element(Index i, Index j) const = 0;
    /// Read the value of the element at row i, column j (using 0-based indices). Eigen-compatible API.
    inline SReal operator() (Index i, Index j) const { return element(i,j); }
    /// Resize the matrix and reset all values to 0
    virtual void resize(Index nbRow, Index nbCol) = 0;
    /// Reset all values to 0
    virtual void clear() = 0;
    /// Write the value of the element at row i, column j (using 0-based indices)
    virtual void set(Index i, Index j, double v) = 0;
    /// Add v to the existing value of the element at row i, column j (using 0-based indices)
    virtual void add(Index i, Index j, double v) = 0;
    /*    /// Write the value of the element at row i, column j (using 0-based indices)
        virtual void set(Index i, Index j, float v) { set(i,j,(double)v); }
        /// Add v to the existing value of the element at row i, column j (using 0-based indices)
        virtual void add(Index i, Index j, float v) { add(i,j,(double)v); }
        /// Reset the value of element i,j to 0
    */    virtual void clear(Index i, Index j) { set(i,j,0.0); }
    /// Reset all the values in row i to 0
    virtual void clearRow(Index i) { for (Index j=0,n=colSize(); j<n; ++j) clear(i,j); }
    /// Clears the value of rows imin to imax-1
    virtual void clearRows(Index imin, Index imax) { for (Index i=imin; i<imax; i++) clearRow(i); }
    /// Reset the all values in column j to 0
    virtual void clearCol(Index j) { for (Index i=0,n=rowSize(); i<n; ++i) clear(i,j); }
    /// Clears all the values in columns imin to imax-1
    virtual void clearCols(Index imin, Index imax) { for (Index i=imin; i<imax; i++) clearCol(i); }
    /// Reset the value of both row and column i to 0
    virtual void clearRowCol(Index i) { clearRow(i); clearCol(i); }
    /// Clears all the values in rows imin to imax-1 and columns imin to imax-1
    virtual void clearRowsCols(Index imin, Index imax) { clearRows(imin,imax); clearCols(imin,imax); }
    /** Make the final data setup after adding entries. For most concrete types, this method does nothing.
      */
    virtual void compress();

    /// @name Get information about the content and structure of this matrix (diagonal, band, sparse, full, block size, ...)
    /// @{

    enum ElementType
    {
        ELEMENT_UNKNOWN = 0,
        ELEMENT_FLOAT,
        ELEMENT_INT,
    };

    /// @return type of elements stored in this matrix
    virtual ElementType getElementType() const { return ELEMENT_FLOAT; }

    /// @return size of elements stored in this matrix
    virtual std::size_t getElementSize() const { return sizeof(SReal); }

    enum MatrixCategory
    {
        MATRIX_UNKNOWN = 0,
        MATRIX_IDENTITY,
        MATRIX_DIAGONAL,
        MATRIX_BAND,
        MATRIX_SPARSE,
        MATRIX_FULL,
    };

    /// @return the category of this matrix
    virtual MatrixCategory getCategory() const { return MATRIX_UNKNOWN; }

    /// @return the number of rows in each block, or 1 of there are no fixed block size
    virtual Index getBlockRows() const { return 1; }

    /// @return the number of columns in each block, or 1 of there are no fixed block size
    virtual Index getBlockCols() const { return 1; }

    /// @return the number of rows of blocks
    virtual Index bRowSize() const { return rowSize() / getBlockRows(); }

    /// @return the number of columns of blocks
    virtual Index bColSize() const { return colSize() / getBlockCols(); }

    /// @return the width of the band on each side of the diagonal (only for band matrices)
    virtual Index getBandWidth() const { return -1; }

    /// @return true if this matrix is diagonal
    bool isDiagonal() const
    {
        MatrixCategory cat = getCategory();
        return (cat == MATRIX_IDENTITY)
                || (cat == MATRIX_DIAGONAL && getBlockRows() == 1 && getBlockCols() == 1)
                || (cat == MATRIX_BAND && getBandWidth() == 0);
    }

    /// @return true if this matrix is block-diagonal
    bool isBlockDiagonal() const
    {
        MatrixCategory cat = getCategory();
        return (cat == MATRIX_IDENTITY)
                || (cat == MATRIX_DIAGONAL)
                || (cat == MATRIX_BAND && getBandWidth() == 0);
    }

    /// @return true if this matrix is band
    bool isBand() const
    {
        MatrixCategory cat = getCategory();
        return (cat == MATRIX_IDENTITY)
                || (cat == MATRIX_DIAGONAL)
                || (cat == MATRIX_BAND);
    }

    /// @return true if this matrix is sparse
    bool isSparse() const
    {
        MatrixCategory cat = getCategory();
        return (cat == MATRIX_IDENTITY)
                || (cat == MATRIX_DIAGONAL)
                || (cat == MATRIX_BAND)
                || (cat == MATRIX_SPARSE);
    }

    /// @}

    /// @name Internal data structures for iterators (should only be used by classes deriving from BaseMatrix)
    /// @{

    class InternalBlockAccessor
    {
    public:
        Index row;
        Index col;
        union
        {
            void* ptr;
            Index data;
        };

        InternalBlockAccessor()
            : row(-1), col(-1)
        {
            ptr = NULL;
            data = 0;
        }

        InternalBlockAccessor(Index row, Index col, void* internalPtr)
            : row(row), col(col)
        {
            data = 0;
            ptr = internalPtr;
        }

        InternalBlockAccessor(Index row, Index col, Index internalData)
            : row(row), col(col)
        {
            ptr = NULL;
            data = internalData;
        }
    };

    class InternalColBlockIterator
    {
    public:
        Index row;
        union
        {
            void* ptr;
            Index data;
        };

        InternalColBlockIterator()
            : row(-1)
        {
            ptr = NULL;
            data = 0;
        }

        InternalColBlockIterator(Index row, void* internalPtr)
            : row(row)
        {
            data = 0;
            ptr = internalPtr;
        }

        InternalColBlockIterator(Index row, Index internalData)
            : row(row)
        {
            ptr = NULL;
            data = internalData;
        }
    };

    class InternalRowBlockIterator
    {
    public:
        union
        {
            void* ptr;
            Index data[2];
        };

        InternalRowBlockIterator()
        {
            ptr = NULL;
            data[0] = 0;
            data[1] = 0;
        }

        InternalRowBlockIterator(void* internalPtr)
        {
            data[0] = 0;
            data[1] = 0;
            ptr = internalPtr;
        }

        InternalRowBlockIterator(Index internalData0, Index internalData1)
        {
            ptr = NULL;
            data[0] = internalData0;
            data[1] = internalData1;
        }
    };

    /// @}

    /// @name Virtual iterator classes and methods
    /// @{

public:
    class BlockAccessor;
    class BlockConstAccessor;
    class ColBlockConstIterator;
    class RowBlockConstIterator;

    class BlockAccessor
    {
    protected:
        BaseMatrix* matrix;
        InternalBlockAccessor internal;

        BlockAccessor()
            : matrix(NULL)
        {
        }

        BlockAccessor(BaseMatrix* matrix, Index row, Index col, void* internalPtr)
            : matrix(matrix), internal(row, col, internalPtr)
        {
        }

        BlockAccessor(BaseMatrix* matrix, Index row, Index col, Index internalData)
            : matrix(matrix), internal(row, col, internalData)
        {
        }

    public:
        ~BlockAccessor()
        {
            if (matrix)
                matrix->bAccessorDelete(&internal);
        }

        BlockAccessor(const BlockAccessor& b)
            : matrix(b.matrix), internal(b.internal)
        {
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        void operator=(const BlockAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(&internal);
            matrix = b.matrix; internal = b.internal;
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        Index getRow() const { return internal.row; }

        Index getCol() const { return internal.col; }

        const BaseMatrix* getMatrix() const { return matrix; }

        BaseMatrix* getMatrix() { return matrix; }

        bool isValid() const
        {
            return matrix && internal.row < (Index)matrix->rowSize() && internal.col < (Index)matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(Index i, Index j) const
        {
            return matrix->bAccessorElement(&internal, i, j);
        }

        /// Write the value of the element at row i, column j within this block (using 0-based indices)
        void set(Index i, Index j, double v)
        {
            matrix->bAccessorSet(&internal, i, j, v);
        }

        /// Add v to the existing value of the element at row i, column j within this block (using 0-based indices)
        void add(Index i, Index j, double v)
        {
            matrix->bAccessorAdd(&internal, i, j, v);
        }

        /// Read all values from this bloc into given float buffer, or return the pointer to the data if the in-memory format is compatible
        const float* elements(float* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        /// Read all values from this bloc into given double buffer, or return the pointer to the data if the in-memory format is compatible
        const double* elements(double* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        /// Read all values from this bloc into given int buffer, or return the pointer to the data if the in-memory format is compatible
        const int* elements(int* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        /// Set all values of this bloc from the given float buffer
        void set(const float* src)
        {
            matrix->bAccessorSet(&internal, src);
        }

        /// Set all values of this bloc from the given double buffer
        void set(const double* src)
        {
            matrix->bAccessorSet(&internal, src);
        }

        /// Set all values of this bloc from the given int buffer
        void set(const int* src)
        {
            matrix->bAccessorSet(&internal, src);
        }

        /// Add to all values of this bloc from the given float buffer
        void add(const float* src)
        {
            matrix->bAccessorAdd(&internal, src);
        }

        /// Add to all values of this bloc from the given double buffer
        void add(const double* src)
        {
            matrix->bAccessorAdd(&internal, src);
        }

        /// Add to all values of this bloc from the given int buffer
        void add(const int* src)
        {
            matrix->bAccessorAdd(&internal, src);
        }

        /// Prepare the addition of float values to this bloc.
        /// Return a pointer to a float buffer where values can be added.
        /// If the in-memory format of the matrix is incompatible, the provided buffer can be used,
        /// but the method must clear it before returning.
        const float* prepareAdd(float* buffer)
        {
            return matrix->bAccessorPrepareAdd(&internal, buffer);
        }

        /// Finalize an addition of float values to this bloc.
        /// The buffer must be the one returned by calling the prepareAdd method.
        void finishAdd(const float* buffer)
        {
            matrix->bAccessorFinishAdd(&internal, buffer);
        }

        /// Prepare the addition of double values to this bloc.
        /// Return a pointer to a double buffer where values can be added.
        /// If the in-memory format of the matrix is incompatible, the provided buffer can be used,
        /// but the method must clear it before returning.
        const double* prepareAdd(double* buffer)
        {
            return matrix->bAccessorPrepareAdd(&internal, buffer);
        }

        /// Finalize an addition of double values to this bloc.
        /// The buffer must be the one returned by calling the prepareAdd method.
        void finishAdd(const double* buffer)
        {
            matrix->bAccessorFinishAdd(&internal, buffer);
        }

        /// Prepare the addition of int values to this bloc.
        /// Return a pointer to a int buffer where values can be added.
        /// If the in-memory format of the matrix is incompatible, the provided buffer can be used,
        /// but the method must clear it before returning.
        const int* prepareAdd(int* buffer)
        {
            return matrix->bAccessorPrepareAdd(&internal, buffer);
        }

        /// Finalize an addition of int values to this bloc.
        /// The buffer must be the one returned by calling the prepareAdd method.
        void finishAdd(const int* buffer)
        {
            matrix->bAccessorFinishAdd(&internal, buffer);
        }

        friend class BaseMatrix;
        friend class BlockConstAccessor;
        friend class ColBlockConstIterator;
    };

    class BlockConstAccessor
    {
    protected:
        const BaseMatrix* matrix;
        InternalBlockAccessor internal;

        BlockConstAccessor()
            : matrix(NULL)
        {
        }

        BlockConstAccessor(const BaseMatrix* matrix, Index row, Index col, void* internalPtr)
            : matrix(matrix), internal(row, col, internalPtr)
        {
        }

        BlockConstAccessor(const BaseMatrix* matrix, Index row, Index col, Index internalData)
            : matrix(matrix), internal(row, col, internalData)
        {
        }

    public:
        ~BlockConstAccessor()
        {
            if (matrix)
                matrix->bAccessorDelete(&internal);
        }

        BlockConstAccessor(const BlockConstAccessor& b)
            : matrix(b.matrix), internal(b.internal)
        {
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        BlockConstAccessor(const BlockAccessor& b)
            : matrix(b.matrix), internal(b.internal)
        {
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        void operator=(const BlockConstAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(&internal);
            matrix = b.matrix; internal = b.internal;
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        void operator=(const BlockAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(&internal);
            matrix = b.matrix; internal = b.internal;
            if (matrix)
                matrix->bAccessorCopy(&internal);
        }

        Index getRow() const { return internal.row; }

        Index getCol() const { return internal.col; }

        const BaseMatrix* getMatrix() const { return matrix; }

        bool isValid() const
        {
            return matrix && internal.row < (Index)matrix->rowSize() && internal.col < (Index)matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(Index i, Index j) const
        {
            return matrix->bAccessorElement(&internal, i, j);
        }

        /// Read all values from this bloc into given float buffer, or return the pointer to the buffer data if the in-memory format is compatible
        const float* elements(float* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        /// Read all values from this bloc into given double buffer, or return the pointer to the buffer data if the in-memory format is compatible
        const double* elements(double* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        /// Read all values from this bloc into given int buffer, or return the pointer to the buffer data if the in-memory format is compatible
        const int* elements(int* dest) const
        {
            return matrix->bAccessorElements(&internal, dest);
        }

        friend class BaseMatrix;
        friend class ColBlockConstIterator;
    };

protected:

    virtual void bAccessorDelete(const InternalBlockAccessor* /*b*/) const {}
    virtual void bAccessorCopy(InternalBlockAccessor* /*b*/) const {}
    virtual SReal bAccessorElement(const InternalBlockAccessor* b, Index i, Index j) const
    {
        return element(b->row * getBlockRows() + i, b->col * getBlockCols() + j);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, Index i, Index j, double v)
    {
        set(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, Index i, Index j, double v)
    {
        add(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
    }

    template<class T>
    const T* bAccessorElementsDefaultImpl(const InternalBlockAccessor* b, T* buffer) const
    {
        const Index NL = getBlockRows();
        const Index NC = getBlockCols();
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                buffer[l*NC+c] = (T)bAccessorElement(b, l, c);
        return buffer;
    }
    virtual const float* bAccessorElements(const InternalBlockAccessor* b, float* buffer) const
    {
        return bAccessorElementsDefaultImpl<float>(b, buffer);
    }
    virtual const double* bAccessorElements(const InternalBlockAccessor* b, double* buffer) const
    {
        return bAccessorElementsDefaultImpl<double>(b, buffer);
    }
    virtual const int* bAccessorElements(const InternalBlockAccessor* b, int* buffer) const
    {
        return bAccessorElementsDefaultImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorSetDefaultImpl(InternalBlockAccessor* b, const T* buffer)
    {
        const Index NL = getBlockRows();
        const Index NC = getBlockCols();
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                bAccessorSet(b, l, c, (double)buffer[l*NC+c]);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const float* buffer)
    {
        bAccessorSetDefaultImpl<float>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const double* buffer)
    {
        bAccessorSetDefaultImpl<double>(b, buffer);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, const int* buffer)
    {
        bAccessorSetDefaultImpl<int>(b, buffer);
    }

    template<class T>
    void bAccessorAddDefaultImpl(InternalBlockAccessor* b, const T* buffer)
    {
        const Index NL = getBlockRows();
        const Index NC = getBlockCols();
        for (Index l=0; l<NL; ++l)
            for (Index c=0; c<NC; ++c)
                bAccessorAdd(b, l, c, (double)buffer[l*NC+c]);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const float* buffer)
    {
        bAccessorAddDefaultImpl<float>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const double* buffer)
    {
        bAccessorAddDefaultImpl<double>(b, buffer);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, const int* buffer)
    {
        bAccessorAddDefaultImpl<int>(b, buffer);
    }

    template<class T>
    T* bAccessorPrepareAddDefaultImpl(InternalBlockAccessor* /*b*/, T* buffer)
    {
        const Index NL = getBlockRows();
        const Index NC = getBlockCols();
        const Index N = NL*NC;
        for (Index i=0; i<N; ++i)
            buffer[i] = (T)0;
        return buffer;
    }
    virtual float* bAccessorPrepareAdd(InternalBlockAccessor* b, float* buffer)
    {
        return bAccessorPrepareAddDefaultImpl<float>(b, buffer);
    }
    virtual double* bAccessorPrepareAdd(InternalBlockAccessor* b, double* buffer)
    {
        return bAccessorPrepareAddDefaultImpl<double>(b, buffer);
    }
    virtual int* bAccessorPrepareAdd(InternalBlockAccessor* b, int* buffer)
    {
        return bAccessorPrepareAddDefaultImpl<int>(b, buffer);
    }

    virtual void bAccessorFinishAdd(InternalBlockAccessor* b, const float* buffer)
    {
        bAccessorAdd(b, buffer);
    }
    virtual void bAccessorFinishAdd(InternalBlockAccessor* b, const double* buffer)
    {
        bAccessorAdd(b, buffer);
    }
    virtual void bAccessorFinishAdd(InternalBlockAccessor* b, const int* buffer)
    {
        bAccessorAdd(b, buffer);
    }

    BlockAccessor createBlockAccessor(Index row, Index col, void* internalPtr = NULL)
    {
        return BlockAccessor(this, row, col, internalPtr);
    }

    BlockAccessor createBlockAccessor(Index row, Index col, Index internalData)
    {
        return BlockAccessor(this, row, col, internalData);
    }

    BlockConstAccessor createBlockConstAccessor(Index row, Index col, void* internalPtr = NULL) const
    {
        return BlockConstAccessor(this, row, col, internalPtr);
    }

    BlockConstAccessor createBlockConstAccessor(Index row, Index col, Index internalData) const
    {
        return BlockConstAccessor(this, row, col, internalData);
    }

    void setMatrix(BlockAccessor* b) { b->matrix = this; }
    void setMatrix(BlockConstAccessor* b) const { b->matrix = this; }

    static InternalBlockAccessor* getInternal(BlockConstAccessor* b) { return &(b->internal); }
    static const InternalBlockAccessor* getInternal(const BlockConstAccessor* b) { return &(b->internal); }

    static InternalBlockAccessor* getInternal(BlockAccessor* b) { return &(b->internal); }
    static const InternalBlockAccessor* getInternal(const BlockAccessor* b) { return &(b->internal); }

public:


    /// Get read access to a bloc
    virtual BlockConstAccessor blocGet(Index i, Index j) const
    {
        return createBlockConstAccessor(i, j);
    }

    /// Get write access to a bloc
    virtual BlockAccessor blocGetW(Index i, Index j)
    {
        return createBlockAccessor(i, j);
    }

    /// Get write access to a bloc, possibly creating it
    virtual BlockAccessor blocCreate(Index i, Index j)
    {
        return createBlockAccessor(i, j);
    }

    /// Shortcut for blocGet(i,j).elements(buffer)
    template<class T>
    const T* blocElements(Index i, Index j, T* buffer) const
    {
        return blocGet(i,j).elements(buffer);
    }

    /// Shortcut for blocCreate(i,j).set(buffer)
    template<class T>
    void blocSet(Index i, Index j, const T* buffer)
    {
        blocCreate(i,j).set(buffer);
    }

    /// Shortcut for blocCreate(i,j).add(buffer)
    template<class T>
    void blocAdd(Index i, Index j, const T* buffer)
    {
        blocCreate(i,j).add(buffer);
    }

    class ColBlockConstIterator
    {
    protected:
        const BaseMatrix* matrix;
        InternalColBlockIterator internal;
        BlockConstAccessor b;

        ColBlockConstIterator(const BaseMatrix* matrix, Index row, void* internalPtr)
            : matrix(matrix), internal(row, internalPtr)
        {
        }

        ColBlockConstIterator(const BaseMatrix* matrix, Index row, Index internalData)
            : matrix(matrix), internal(row, internalData)
        {
        }

    public:

        ColBlockConstIterator()
            : matrix(NULL)
        {
        }

        ColBlockConstIterator(const ColBlockConstIterator& it2)
            : matrix(it2.matrix), internal(it2.internal), b(it2.b)
        {
            if (matrix)
                matrix->itCopyColBlock(&internal);
        }

        ~ColBlockConstIterator()
        {
            if (matrix)
                matrix->itDeleteColBlock(&internal);
        }

        void operator=(const ColBlockConstIterator& it2)
        {
            if (matrix)
                matrix->itDeleteColBlock(&internal);
            matrix = it2.matrix; internal = it2.internal;
            if (matrix)
                matrix->itCopyColBlock(&internal);
        }

        const BlockConstAccessor& bloc()
        {
            matrix->itAccessColBlock(&internal, &b);
            return b;
        }

        const BlockConstAccessor& operator*()
        {
            return bloc();
        }
        const BlockConstAccessor& operator->()
        {
            return bloc();
        }
        void operator++() // prefix
        {
            matrix->itIncColBlock(&internal);
        }
        void operator++(int) // postfix
        {
            matrix->itIncColBlock(&internal);
        }
        void operator--() // prefix
        {
            matrix->itDecColBlock(&internal);
        }
        void operator--(int) // postfix
        {
            matrix->itDecColBlock(&internal);
        }
        bool operator==(const ColBlockConstIterator& it2) const
        {
            return matrix->itEqColBlock(&internal, &it2.internal);
        }
        bool operator!=(const ColBlockConstIterator& it2) const
        {
            return ! matrix->itEqColBlock(&internal, &it2.internal);
        }
        bool operator<(const ColBlockConstIterator& it2) const
        {
            return matrix->itLessColBlock(&internal, &it2.internal);
        }
        bool operator>(const ColBlockConstIterator& it2) const
        {
            return matrix->itLessColBlock(&it2.internal, &internal);
        }

        friend class BaseMatrix;

    };

protected:

    static InternalColBlockIterator* getInternal(ColBlockConstIterator* b) { return &(b->internal); }
    static const InternalColBlockIterator* getInternal(const ColBlockConstIterator* b) { return &(b->internal); }

    virtual void itCopyColBlock(InternalColBlockIterator* /*it*/) const {}
    virtual void itDeleteColBlock(const InternalColBlockIterator* /*it*/) const {}
    virtual void itAccessColBlock(InternalColBlockIterator* it, BlockConstAccessor* b) const
    {
        setMatrix(b);
        getInternal(b)->row = it->row;
        getInternal(b)->col = it->data;
        getInternal(b)->ptr = NULL;
    }
    virtual void itIncColBlock(InternalColBlockIterator* it) const
    {
        Index col = it->data;
        ++col;
        it->data = col;
    }
    virtual void itDecColBlock(InternalColBlockIterator* it) const
    {
        Index col = it->data;
        --col;
        it->data = col;
    }
    virtual bool itEqColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        Index col = it->data;
        Index col2 = it2->data;
        return col == col2;
    }
    virtual bool itLessColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        Index col = it->data;
        Index col2 = it2->data;
        return col < col2;
    }

    ColBlockConstIterator createColBlockConstIterator(Index row, void* internalPtr) const
    {
        return ColBlockConstIterator(this, row, internalPtr);
    }
    ColBlockConstIterator createColBlockConstIterator(Index row, Index internalData) const
    {
        return ColBlockConstIterator(this, row, internalData);
    }

public:

    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(Index ib) const
    {
        return createColBlockConstIterator(ib, (Index)(0));
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(Index ib) const
    {
        return createColBlockConstIterator(ib, bColSize());
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(Index ib) const
    {
        return std::make_pair(bRowBegin(ib), bRowEnd(ib));
    }


    class RowBlockConstIterator
    {
    protected:
        const BaseMatrix* matrix;
        InternalRowBlockIterator internal;

        RowBlockConstIterator(const BaseMatrix* matrix, void* internalPtr)
            : matrix(matrix), internal(internalPtr)
        {
        }

        RowBlockConstIterator(const BaseMatrix* matrix, Index internalData0, Index internalData1)
            : matrix(matrix), internal(internalData0, internalData1)
        {
        }

    public:

        RowBlockConstIterator()
            : matrix(NULL)
        {
        }

        RowBlockConstIterator(const RowBlockConstIterator& it2)
            : matrix(it2.matrix), internal(it2.internal)
        {
            if (matrix)
                matrix->itCopyRowBlock(&internal);
        }

        ~RowBlockConstIterator()
        {
            if (matrix)
                matrix->itDeleteRowBlock(&internal);
        }

        void operator=(const RowBlockConstIterator& it2)
        {
            if (matrix)
                matrix->itDeleteRowBlock(&internal);
            matrix = it2.matrix; internal = it2.internal;
            if (matrix)
                matrix->itCopyRowBlock(&internal);
        }

        Index row()
        {
            return matrix->itAccessRowBlock(&internal);
        }
        Index operator*()
        {
            return row();
        }

        ColBlockConstIterator begin()
        {
            return matrix->itBeginRowBlock(&internal);
        }

        ColBlockConstIterator end()
        {
            return matrix->itEndRowBlock(&internal);
        }

        std::pair<ColBlockConstIterator,ColBlockConstIterator> range()
        {
            return matrix->itRangeRowBlock(&internal);
        }
        std::pair<ColBlockConstIterator,ColBlockConstIterator> operator->()
        {
            return range();
        }

        void operator++() // prefix
        {
            matrix->itIncRowBlock(&internal);
        }
        void operator++(int) // postfix
        {
            matrix->itIncRowBlock(&internal);
        }
        void operator--() // prefix
        {
            matrix->itDecRowBlock(&internal);
        }
        void operator--(int) // postfix
        {
            matrix->itDecRowBlock(&internal);
        }
        bool operator==(const RowBlockConstIterator& it2) const
        {
            return matrix->itEqRowBlock(&internal, &it2.internal);
        }
        bool operator!=(const RowBlockConstIterator& it2) const
        {
            return ! matrix->itEqRowBlock(&internal, &it2.internal);
        }
        bool operator<(const RowBlockConstIterator& it2) const
        {
            return matrix->itLessRowBlock(&internal, &it2.internal);
        }
        bool operator>(const RowBlockConstIterator& it2) const
        {
            return matrix->itLessRowBlock(&it2.internal, &internal);
        }

        friend class BaseMatrix;
    };

protected:

    static InternalRowBlockIterator* getInternal(RowBlockConstIterator* b) { return &(b->internal); }
    static const InternalRowBlockIterator* getInternal(const RowBlockConstIterator* b) { return &(b->internal); }

    virtual void itCopyRowBlock(InternalRowBlockIterator* /*it*/) const {}
    virtual void itDeleteRowBlock(const InternalRowBlockIterator* /*it*/) const {}
    virtual Index itAccessRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        return row;
    }
    virtual ColBlockConstIterator itBeginRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        return bRowBegin(row);
    }
    virtual ColBlockConstIterator itEndRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        return bRowEnd(row);
    }
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> itRangeRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        return bRowRange(row);
    }

    virtual void itIncRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        ++row;
        it->data[0] = row;
    }
    virtual void itDecRowBlock(InternalRowBlockIterator* it) const
    {
        Index row = (it->data[0]);
        --row;
        it->data[0] = row;
    }
    virtual bool itEqRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        Index row = (it->data[0]);
        Index row2 = (it2->data[0]);
        return row == row2;
    }
    virtual bool itLessRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        Index row = (it->data[0]);
        Index row2 = (it2->data[0]);
        return row < row2;
    }

    RowBlockConstIterator createRowBlockConstIterator(void* internalPtr) const
    {
        return RowBlockConstIterator(this, internalPtr);
    }
    RowBlockConstIterator createRowBlockConstIterator(Index internalData0, Index internalData1) const
    {
        return RowBlockConstIterator(this, internalData0, internalData1);
    }

public:

    /// Get the iterator corresponding to the beginning of the rows of blocks
    virtual RowBlockConstIterator bRowsBegin() const
    {
        return createRowBlockConstIterator(0, 0);
    }

    /// Get the iterator corresponding to the end of the rows of blocks
    virtual RowBlockConstIterator bRowsEnd() const
    {
        return createRowBlockConstIterator(bRowSize(), 0);
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<RowBlockConstIterator, RowBlockConstIterator> bRowsRange() const
    {
        return std::make_pair(bRowsBegin(), bRowsEnd());
    }

    /// @}

    /// @name basic linear operations
    /// @{

public:

    /// Multiply the matrix by vector v and put the result in vector result
    virtual void opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const;

    /// Multiply the matrix by float vector v and put the result in vector result
    virtual void opMulV(float* result, const float* v) const;

    /// Multiply the matrix by double vector v and put the result in vector result
    virtual void opMulV(double* result, const double* v) const;

    /// Multiply the matrix by vector v and add the result in vector result
    virtual void opPMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const;

    /// Multiply the matrix by float vector v and add the result in vector result
    virtual void opPMulV(float* result, const float* v) const;

    /// Multiply the matrix by double vector v and add the result in vector result
    virtual void opPMulV(double* result, const double* v) const;


    /// Multiply the transposed matrix by vector v and put the result in vector result
    virtual void opMulTV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const;

    /// Multiply the transposed matrix by float vector v and put the result in vector result
    virtual void opMulTV(float* result, const float* v) const;

    /// Multiply the transposed matrix by double vector v and put the result in vector result
    virtual void opMulTV(double* result, const double* v) const;

    /// Multiply the transposed matrix by vector v and add the result in vector result
    virtual void opPMulTV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const;

    /// Multiply the transposed matrix by float vector v and add the result in vector result
    virtual void opPMulTV(float* result, const float* v) const;

    /// Multiply the transposed matrix by double vector v and add the result in vector result
    virtual void opPMulTV(double* result, const double* v) const;

    /// Multiply the transposed matrix by matrix m and store the result in matrix result
    virtual void opMulTM(BaseMatrix * result,BaseMatrix * m) const;

    /// Subtract the matrix to the m matrix and strore the result in m
    virtual void opAddM(defaulttype::BaseMatrix* m,double fact) const;

    /// Subtract the transposed matrix to the m matrix and strore the result in m
    virtual void opAddMT(defaulttype::BaseMatrix* m,double fact) const;

    /// @}

    friend std::ostream& operator<<(std::ostream& out, const  sofa::defaulttype::BaseMatrix& m )
    {
        Index nx = m.colSize();
        Index ny = m.rowSize();
        out << "[";
        for (Index y=0; y<ny; ++y)
        {
            out << "\n[";
            for (Index x=0; x<nx; ++x)
            {
                out << " " << m.element(y,x);
            }
            out << " ]";
        }
        out << " ]";
        return out;
    }

    friend std::istream& operator>>( std::istream& in, sofa::defaulttype::BaseMatrix& m )
    {
        // The reading could be way simplier with an other format,
        // but I did not want to change the existing output.
        // Anyway, I guess there are better ways to perform the reading
        // but at least this one is working...

        std::vector<SReal> line;
        std::vector< std::vector<SReal> > lines;

    //    unsigned l=0, c;

        in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the matrix

        while(true)
        {
            in.ignore(INT_MAX, '['); // ignores all characters until it passes a [, start of the line
    //        c=0;

            SReal r;
            char car; in >> car;
            while( car!=']') // end of the line
            {
                in.seekg( -1, std::istream::cur ); // unread car
                in >> r;
                line.push_back(r);
    //            ++c;
                in >> car;
            }

    //        ++l;

            lines.push_back(line);
            line.clear();

            in >> car;
            if( car==']' ) break; // end of the matrix
            else in.seekg( -1, std::istream::cur ); // unread car

        }

        m.resize( (Index)lines.size(), (Index)lines[0].size() );

        for( size_t i=0; i<lines.size();++i)
        {
            assert( lines[i].size() == lines[0].size() ); // all line should have the same number of columns
            for( size_t j=0; j<lines[i].size();++j)
            {
                m.add( (Index)i, (Index)j, lines[i][j] );
            }
        }

        m.compress();


        if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
        return in;
    }

};


} // nampespace defaulttype

} // nampespace sofa

#endif
