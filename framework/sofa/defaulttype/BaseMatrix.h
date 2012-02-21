/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_BASEMATRIX_H
#define SOFA_DEFAULTTYPE_BASEMATRIX_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/defaulttype.h>
#include <sofa/defaulttype/BaseVector.h>
#include <utility> // for std::pair
#include <cstddef> // for NULL
#include <iostream>

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
    BaseMatrix();
    virtual ~BaseMatrix();

    /// Number of rows
    virtual unsigned int rowSize(void) const = 0;
    /// Number of columns
    virtual unsigned int colSize(void) const = 0;
    /// Read the value of the element at row i, column j (using 0-based indices)
    virtual SReal element(int i, int j) const = 0;
    /// Resize the matrix and reset all values to 0
    virtual void resize(int nbRow, int nbCol) = 0;
    /// Reset all values to 0
    virtual void clear() = 0;
    /// Write the value of the element at row i, column j (using 0-based indices)
    virtual void set(int i, int j, double v) = 0;
    /// Add v to the existing value of the element at row i, column j (using 0-based indices)
    virtual void add(int i, int j, double v) = 0;
    /*    /// Write the value of the element at row i, column j (using 0-based indices)
        virtual void set(int i, int j, float v) { set(i,j,(double)v); }
        /// Add v to the existing value of the element at row i, column j (using 0-based indices)
        virtual void add(int i, int j, float v) { add(i,j,(double)v); }
        /// Reset the value of element i,j to 0
    */    virtual void clear(int i, int j) { set(i,j,0.0); }
    /// Reset the value of row i to 0
    virtual void clearRow(int i) { for (int j=0,n=colSize(); j<n; ++j) clear(i,j); }
    /// Reset the value of column j to 0
    virtual void clearCol(int j) { for (int i=0,n=rowSize(); i<n; ++i) clear(i,j); }
    /// Reset the value of both row and column i to 0
    virtual void clearRowCol(int i) { clearRow(i); clearCol(i); }

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
    virtual unsigned int getElementSize() const { return sizeof(SReal); }

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
    virtual int getBlockRows() const { return 1; }

    /// @return the number of columns in each block, or 1 of there are no fixed block size
    virtual int getBlockCols() const { return 1; }

    /// @return the number of rows of blocks
    virtual int bRowSize() const { return rowSize() / getBlockRows(); }

    /// @return the number of columns of blocks
    virtual int bColSize() const { return colSize() / getBlockCols(); }

    /// @return the width of the band on each side of the diagonal (only for band matrices)
    virtual int getBandWidth() const { return -1; }

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
        int row;
        int col;
        union
        {
            void* ptr;
            int data;
        };

        InternalBlockAccessor()
            : row(-1), col(-1)
        {
            ptr = NULL;
            data = 0;
        }

        InternalBlockAccessor(int row, int col, void* internalPtr)
            : row(row), col(col)
        {
            data = 0;
            ptr = internalPtr;
        }

        InternalBlockAccessor(int row, int col, int internalData)
            : row(row), col(col)
        {
            ptr = NULL;
            data = internalData;
        }
    };

    class InternalColBlockIterator
    {
    public:
        int row;
        union
        {
            void* ptr;
            int data;
        };

        InternalColBlockIterator()
            : row(-1)
        {
            ptr = NULL;
            data = 0;
        }

        InternalColBlockIterator(int row, void* internalPtr)
            : row(row)
        {
            data = 0;
            ptr = internalPtr;
        }

        InternalColBlockIterator(int row, int internalData)
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
            int data[2];
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

        InternalRowBlockIterator(int internalData0, int internalData1)
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

        BlockAccessor(BaseMatrix* matrix, int row, int col, void* internalPtr)
            : matrix(matrix), internal(row, col, internalPtr)
        {
        }

        BlockAccessor(BaseMatrix* matrix, int row, int col, int internalData)
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

        int getRow() const { return internal.row; }

        int getCol() const { return internal.col; }

        const BaseMatrix* getMatrix() const { return matrix; }

        BaseMatrix* getMatrix() { return matrix; }

        bool isValid() const
        {
            return matrix && (unsigned) internal.row < matrix->rowSize() && (unsigned) internal.col < matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(int i, int j) const
        {
            return matrix->bAccessorElement(&internal, i, j);
        }

        /// Write the value of the element at row i, column j within this block (using 0-based indices)
        void set(int i, int j, double v)
        {
            matrix->bAccessorSet(&internal, i, j, v);
        }

        /// Add v to the existing value of the element at row i, column j within this block (using 0-based indices)
        void add(int i, int j, double v)
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

        BlockConstAccessor(const BaseMatrix* matrix, int row, int col, void* internalPtr)
            : matrix(matrix), internal(row, col, internalPtr)
        {
        }

        BlockConstAccessor(const BaseMatrix* matrix, int row, int col, int internalData)
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

        int getRow() const { return internal.row; }

        int getCol() const { return internal.col; }

        const BaseMatrix* getMatrix() const { return matrix; }

        bool isValid() const
        {
            return matrix && (unsigned) internal.row < matrix->rowSize() && (unsigned) internal.col < matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(int i, int j) const
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
    virtual SReal bAccessorElement(const InternalBlockAccessor* b, int i, int j) const
    {
        return element(b->row * getBlockRows() + i, b->col * getBlockCols() + j);
    }
    virtual void bAccessorSet(InternalBlockAccessor* b, int i, int j, double v)
    {
        set(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
    }
    virtual void bAccessorAdd(InternalBlockAccessor* b, int i, int j, double v)
    {
        add(b->row * getBlockRows() + i, b->col * getBlockCols() + j, v);
    }

    template<class T>
    const T* bAccessorElementsDefaultImpl(const InternalBlockAccessor* b, T* buffer) const
    {
        const int NL = getBlockRows();
        const int NC = getBlockCols();
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
        const int NL = getBlockRows();
        const int NC = getBlockCols();
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
        const int NL = getBlockRows();
        const int NC = getBlockCols();
        for (int l=0; l<NL; ++l)
            for (int c=0; c<NC; ++c)
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
        const int NL = getBlockRows();
        const int NC = getBlockCols();
        const int N = NL*NC;
        for (int i=0; i<N; ++i)
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

    BlockAccessor createBlockAccessor(int row, int col, void* internalPtr = NULL)
    {
        return BlockAccessor(this, row, col, internalPtr);
    }

    BlockAccessor createBlockAccessor(int row, int col, int internalData)
    {
        return BlockAccessor(this, row, col, internalData);
    }

    BlockConstAccessor createBlockConstAccessor(int row, int col, void* internalPtr = NULL) const
    {
        return BlockConstAccessor(this, row, col, internalPtr);
    }

    BlockConstAccessor createBlockConstAccessor(int row, int col, int internalData) const
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
    virtual BlockConstAccessor blocGet(int i, int j) const
    {
        return createBlockConstAccessor(i, j);
    }

    /// Get write access to a bloc
    virtual BlockAccessor blocGetW(int i, int j)
    {
        return createBlockAccessor(i, j);
    }

    /// Get write access to a bloc, possibly creating it
    virtual BlockAccessor blocCreate(int i, int j)
    {
        return createBlockAccessor(i, j);
    }

    /// Shortcut for blocGet(i,j).elements(buffer)
    template<class T>
    const T* blocElements(int i, int j, T* buffer) const
    {
        return blocGet(i,j).elements(buffer);
    }

    /// Shortcut for blocCreate(i,j).set(buffer)
    template<class T>
    void blocSet(int i, int j, const T* buffer)
    {
        blocCreate(i,j).set(buffer);
    }

    /// Shortcut for blocCreate(i,j).add(buffer)
    template<class T>
    void blocAdd(int i, int j, const T* buffer)
    {
        blocCreate(i,j).add(buffer);
    }

    class ColBlockConstIterator
    {
    protected:
        const BaseMatrix* matrix;
        InternalColBlockIterator internal;
        BlockConstAccessor b;

        ColBlockConstIterator(const BaseMatrix* matrix, int row, void* internalPtr)
            : matrix(matrix), internal(row, internalPtr)
        {
        }

        ColBlockConstIterator(const BaseMatrix* matrix, int row, int internalData)
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
        int col = it->data;
        ++col;
        it->data = col;
    }
    virtual void itDecColBlock(InternalColBlockIterator* it) const
    {
        int col = it->data;
        --col;
        it->data = col;
    }
    virtual bool itEqColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        int col = it->data;
        int col2 = it2->data;
        return col == col2;
    }
    virtual bool itLessColBlock(const InternalColBlockIterator* it, const InternalColBlockIterator* it2) const
    {
        int col = it->data;
        int col2 = it2->data;
        return col < col2;
    }

    ColBlockConstIterator createColBlockConstIterator(int row, void* internalPtr) const
    {
        return ColBlockConstIterator(this, row, internalPtr);
    }
    ColBlockConstIterator createColBlockConstIterator(int row, int internalData) const
    {
        return ColBlockConstIterator(this, row, internalData);
    }

public:

    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(int ib) const
    {
        return createColBlockConstIterator(ib, (int)(0));
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(int ib) const
    {
        return createColBlockConstIterator(ib, (int)(bColSize()));
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(int ib) const
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

        RowBlockConstIterator(const BaseMatrix* matrix, int internalData0, int internalData1)
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

        int row()
        {
            return matrix->itAccessRowBlock(&internal);
        }
        int operator*()
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
    virtual int itAccessRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        return row;
    }
    virtual ColBlockConstIterator itBeginRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        return bRowBegin(row);
    }
    virtual ColBlockConstIterator itEndRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        return bRowEnd(row);
    }
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> itRangeRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        return bRowRange(row);
    }

    virtual void itIncRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        ++row;
        it->data[0] = row;
    }
    virtual void itDecRowBlock(InternalRowBlockIterator* it) const
    {
        int row = (it->data[0]);
        --row;
        it->data[0] = row;
    }
    virtual bool itEqRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        int row = (it->data[0]);
        int row2 = (it2->data[0]);
        return row == row2;
    }
    virtual bool itLessRowBlock(const InternalRowBlockIterator* it, const InternalRowBlockIterator* it2) const
    {
        int row = (it->data[0]);
        int row2 = (it2->data[0]);
        return row < row2;
    }

    RowBlockConstIterator createRowBlockConstIterator(void* internalPtr) const
    {
        return RowBlockConstIterator(this, internalPtr);
    }
    RowBlockConstIterator createRowBlockConstIterator(int internalData0, int internalData1) const
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

    friend std::ostream& operator << (std::ostream& out, const  sofa::defaulttype::BaseMatrix& v )
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

};


} // nampespace defaulttype

} // nampespace sofa

#endif
