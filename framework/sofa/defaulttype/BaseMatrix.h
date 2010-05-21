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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_BASEMATRIX_H
#define SOFA_DEFAULTTYPE_BASEMATRIX_H

#include <sofa/helper/system/config.h>
#include <utility> // for std::pair

namespace sofa
{

namespace defaulttype
{

/// Generic matrix API, allowing to fill and use a matrix independently of the linear algebra library in use.
///
/// Note that accessing values using this class is rather slow and should only be used in codes where the
/// provided genericity is necessary.
class BaseMatrix
{
public:
    virtual ~BaseMatrix() {}

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
    virtual unsigned int getElementSize() const { return sizeof(double); }

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


    /// @name Virtual iterator classes and methods
    /// @{

    class BaseBlockAccessor;
    class BlockAccessor;
    class BlockConstAccessor;
    class ColBlockConstIterator;

    class BaseBlockAccessor
    {
    protected:
        int row;
        int col;
        void* internalData;

        BaseBlockAccessor()
            : row(-1), col(-1), internalData(NULL)
        {
        }

        BaseBlockAccessor(int row, int col, void* internalData)
            : row(row), col(col), internalData(internalData)
        {
        }

    public:

        int getRow() const { return row; }

        int getCol() const { return col; }

        friend class BaseMatrix;
        friend class ColBlockConstIterator;
    };

    virtual void bAccessorDelete(const BaseBlockAccessor* /*b*/) const {}
    virtual void bAccessorCopy(BaseBlockAccessor* /*b*/) const {}
    virtual SReal bAccessorElement(const BaseBlockAccessor* b, int i, int j) const
    {
        return element(b->getRow() * getBlockRows() + i, b->getCol() * getBlockCols() + j);
    }
    virtual void bAccessorSet(BaseBlockAccessor* b, int i, int j, double v)
    {
        set(b->getRow() * getBlockRows() + i, b->getCol() * getBlockCols() + j, v);
    }
    virtual void bAccessorAdd(BaseBlockAccessor* b, int i, int j, double v)
    {
        add(b->getRow() * getBlockRows() + i, b->getCol() * getBlockCols() + j, v);
    }

    class BlockAccessor : public BaseBlockAccessor
    {
    protected:
        BaseMatrix* matrix;

        BlockAccessor()
            : matrix(NULL)
        {
        }

        BlockAccessor(BaseMatrix* matrix, int row, int col, void* internalData)
            : BaseBlockAccessor(row, col, internalData), matrix(matrix)
        {
        }

    public:
        ~BlockAccessor()
        {
            if (matrix)
                matrix->bAccessorDelete(this);
        }

        BlockAccessor(const BlockAccessor& b)
            : BaseBlockAccessor(b.row, b.col, b.internalData), matrix(b.matrix)
        {
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        void operator=(const BlockAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(this);
            matrix = b.matrix; row = b.row; col = b.col; internalData = b.internalData;
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        const BaseMatrix* getMatrix() const { return matrix; }

        BaseMatrix* getMatrix() { return matrix; }

        bool isValid() const
        {
            return matrix && (unsigned) row < matrix->rowSize() && (unsigned) col < matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(int i, int j) const
        {
            return matrix->bAccessorElement(this, i, j);
        }

        /// Write the value of the element at row i, column j within this block (using 0-based indices)
        void set(int i, int j, double v)
        {
            matrix->bAccessorSet(this, i, j, v);
        }

        /// Add v to the existing value of the element at row i, column j within this block (using 0-based indices)
        void add(int i, int j, double v)
        {
            matrix->bAccessorAdd(this, i, j, v);
        }

        friend class BaseMatrix;
        friend class BlockConstAccessor;
        friend class ColBlockConstIterator;
    };

    class BlockConstAccessor : public BaseBlockAccessor
    {
    protected:
        const BaseMatrix* matrix;

        BlockConstAccessor()
            : matrix(NULL)
        {
        }

        BlockConstAccessor(const BaseMatrix* matrix, int row, int col, void* internalData)
            : BaseBlockAccessor(row, col, internalData), matrix(matrix)
        {
        }

    public:
        ~BlockConstAccessor()
        {
            if (matrix)
                matrix->bAccessorDelete(this);
        }

        BlockConstAccessor(const BlockConstAccessor& b)
            : BaseBlockAccessor(b.row, b.col, b.internalData), matrix(b.matrix)
        {
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        BlockConstAccessor(const BlockAccessor& b)
            : BaseBlockAccessor(b.row, b.col, b.internalData), matrix(b.matrix)
        {
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        void operator=(const BlockConstAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(this);
            matrix = b.matrix; row = b.row; col = b.col; internalData = b.internalData;
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        void operator=(const BlockAccessor& b)
        {
            if (matrix)
                matrix->bAccessorDelete(this);
            matrix = b.matrix; row = b.row; col = b.col; internalData = b.internalData;
            if (matrix)
                matrix->bAccessorCopy(this);
        }

        const BaseMatrix* getMatrix() const { return matrix; }

        bool isValid() const
        {
            return matrix && (unsigned) row < matrix->rowSize() && (unsigned) col < matrix->colSize();
        }

        /// Read the value of the element at row i, column j within this block (using 0-based indices)
        SReal element(int i, int j) const
        {
            return matrix->bAccessorElement(this, i, j);
        }

        friend class BaseMatrix;
        friend class ColBlockConstIterator;
    };

    class ColBlockConstIterator
    {
    protected:
        const BaseMatrix* matrix;
        int row;
        void* internalData;
        BlockConstAccessor b;

        ColBlockConstIterator(const BaseMatrix* matrix, int row, void* internalData)
            : matrix(matrix), row(row), internalData(internalData)
        {
        }

    public:

        ColBlockConstIterator()
            : matrix(NULL), row(-1), internalData(NULL)
        {
        }

        ColBlockConstIterator(const ColBlockConstIterator& it2)
            : matrix(it2.matrix), row(it2.row), internalData(it2.internalData), b(it2.b)
        {
            if (matrix)
                matrix->itCopyColBlockConstIterator(this);
        }

        ~ColBlockConstIterator()
        {
            if (matrix)
                matrix->itDeleteColBlockConstIterator(this);
        }

        const BlockConstAccessor& bloc()
        {
            matrix->itAccessColBlockConstIterator(this, &b);
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
            matrix->itIncColBlockConstIterator(this);
        }
        void operator++(int) // postfix
        {
            matrix->itIncColBlockConstIterator(this);
        }
        void operator--() // prefix
        {
            matrix->itIncColBlockConstIterator(this);
        }
        void operator--(int) // postfix
        {
            matrix->itDecColBlockConstIterator(this);
        }
        bool operator==(const ColBlockConstIterator& it2) const
        {
            return matrix->itEqColBlockConstIterator(this, &it2);
        }
        bool operator!=(const ColBlockConstIterator& it2) const
        {
            return ! matrix->itEqColBlockConstIterator(this, &it2);
        }
        bool operator<(const ColBlockConstIterator& it2) const
        {
            return matrix->itLessColBlockConstIterator(this, &it2);
        }
        bool operator>(const ColBlockConstIterator& it2) const
        {
            return matrix->itLessColBlockConstIterator(&it2, this);
        }

        friend class BaseMatrix;

    };

    virtual void itCopyColBlockConstIterator(ColBlockConstIterator* /*it*/) const {}
    virtual void itDeleteColBlockConstIterator(const ColBlockConstIterator* /*it*/) const {}
    virtual void itAccessColBlockConstIterator(ColBlockConstIterator* it, BlockConstAccessor* b) const
    {
        b->matrix = this;
        b->row = it->row;
        b->col = (int)(long)(it->internalData);
        b->internalData = NULL;
    }
    virtual void itIncColBlockConstIterator(ColBlockConstIterator* it) const
    {
        int col = (int)(long)(it->internalData);
        ++col;
        it->internalData = (void*)(long)col;
    }
    virtual void itDecColBlockConstIterator(ColBlockConstIterator* it) const
    {
        int col = (int)(long)(it->internalData);
        ++col;
        it->internalData = (void*)(long)col;
    }
    virtual bool itEqColBlockConstIterator(const ColBlockConstIterator* it, const ColBlockConstIterator* it2) const
    {
        int col = (int)(long)(it->internalData);
        int col2 = (int)(long)(it2->internalData);
        return col == col2;
    }
    virtual bool itLessColBlockConstIterator(const ColBlockConstIterator* it, const ColBlockConstIterator* it2) const
    {
        int col = (int)(long)(it->internalData);
        int col2 = (int)(long)(it2->internalData);
        return col < col2;
    }

    /// Get the iterator corresponding to the beginning of the given row of blocks
    virtual ColBlockConstIterator bRowBegin(int bi)
    {
        return ColBlockConstIterator(this, bi, (void*)(long)(0));
    }

    /// Get the iterator corresponding to the end of the given row of blocks
    virtual ColBlockConstIterator bRowEnd(int bi)
    {
        return ColBlockConstIterator(this, bi, (void*)(long)(bColSize()));
    }

    /// Get the iterators corresponding to the beginning and end of the given row of blocks
    virtual std::pair<ColBlockConstIterator, ColBlockConstIterator> bRowRange(int bi)
    {
        return std::make_pair(bRowBegin(bi), bRowEnd(bi));
    }

    /*
        class RowBlockConstIterator
        {
        public:

        };
        class BlockConstIterator
        {
        public:
        };
    */

    /// @}

};


} // nampespace defaulttype

} // nampespace sofa

#endif
