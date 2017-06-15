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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenBaseSparseMatrix_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenBaseSparseMatrix_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/SortedPermutation.h>
#include <sofa/helper/vector.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <map>
#include <Eigen/Sparse>

#ifdef _OPENMP
#include "EigenBaseSparseMatrix_MT.h"
#endif




namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define EigenBaseSparseMatrix_CHECK
//#define EigenBaseSparseMatrix_VERBOSE






/** Sparse matrix based on the Eigen library.

An Eigen::SparseMatrix<Real, RowMajor> matrix is used to store the data in Compressed Row Storage mode.
This matrix can not be accessed randomly. Two access modes are implemented.

The first access mode consists in inserting entries in increasing row, increasing column order.
Method beginRow(Index index) must be called before any entry can be appended to row i.
@warning beginRow must be called even for empty rows
Then insertBack(i,j,value) must be used in for increasing j. There is no need to explicitly end a row.
Finally, method compress() must be called after the last entry has been inserted.
This is the most efficient access mode.

The second access mode is randow access, but you access an auxiliary matrix.
Method add is used to add a value at a given location.
Method compress() is then used to transfer this data to the compressed matrix.
There is no way to replace an entry, you can only add.

Rows, columns, or the full matrix can be set to zero using the clear* methods.

  */
template<class TReal>
class EigenBaseSparseMatrix : public defaulttype::BaseMatrix
{
    void set(Index i, Index j, double v)
    {
        for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
        {
            if(it.index()==j) // diagonal entry
                it.valueRef()=0.0;
        }

        incoming.push_back( Triplet(i,j,(Real)v) );
    }


public:

    typedef TReal Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef EigenBaseSparseMatrix<TReal> ThisMatrix;
    typedef Eigen::Triplet<Real> Triplet;

private:

    helper::vector<Triplet> incoming;             ///< Scheduled additions


public:

    CompressedMatrix compressedMatrix;    ///< the compressed matrix


    EigenBaseSparseMatrix(Index nbRow=0, Index nbCol=0)
    {
        resize(nbRow,nbCol);
    }

    /// copy constructor
    EigenBaseSparseMatrix(const ThisMatrix& m)
    {
        *this = m;
    }

    /// copy operator
    void operator=(const ThisMatrix& m)
    {
        incoming = m.incoming;
        compressedMatrix = m.compressedMatrix;
    }

    void setIdentity()
    {
        clear();
        for( Index i=0; i<rowSize(); i++ )
        {
            if( i==colSize() ) break;
            add(i,i,1.0);
        }
        compress();
    }

    /// Schedule the addition of the value at the given place. Scheduled additions must be finalized using function compress().
    void add( Index row, Index col, double value ){
        if( value!=0.0 ) incoming.push_back( Triplet(row,col,(Real)value) );
    }

    void beginRow(Index index)
    {
        compressedMatrix.startVec(index);
    }

    /// Insert in the compressed matrix. There must be no value at this place already. Efficient only if the value is inserted at the last place of the last row.
    /// @warning the line must be created previously with "beginRow"
    void insertBack( Index row, Index col, Real value){
        if( value!=0.0 ) compressedMatrix.insertBack(row,col) = value;
    }

    /// Return a reference to the given entry in the compressed matrix.There can (must ?) be a value at this place already. Efficient only if the it is at the last place of the compressed matrix.
    Real& coeffRef( Index i, Index j ){
        return compressedMatrix.coeffRef(i,j);
    }

    /** Clear and resize this to (m.rows,nbCol) and initialize it with the given matrix, columns shifted of the given value: this(i,j+shift) = m(i,j).
      @precond nbCol >= m.cols + shift
      */
    void copy(const EigenBaseSparseMatrix& m, unsigned nbCol, unsigned shift)
    {
        resize(m.rowSize(),nbCol);
        const CompressedMatrix& im = m.compressedMatrix;
        for(Index i=0; i<im.rows(); i++)
        {
            for(typename CompressedMatrix::InnerIterator j(im,i); j; ++j)
                add(i,shift+j.col(),j.value());
        }
        compress();
    }


    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(Index nbRow, Index nbCol)
    {
        compressedMatrix.resize(nbRow,nbCol);
    }



    /// number of rows
    Index rowSize(void) const
    {
        return compressedMatrix.rows();
    }

    /// number of columns
    Index colSize(void) const
    {
        return compressedMatrix.cols();
    }

    inline void reserve(typename CompressedMatrix::Index reserveSize)
    {
        compressedMatrix.reserve(reserveSize);
    }

    SReal element(Index i, Index j) const
    {
        return (SReal)compressedMatrix.coeff(i,j);
    }



    /// Add the values from the scheduled list, and clears the schedule list. @sa set(Index i, Index j, double v).
    void compress()
    {
        if( incoming.empty() ) return;
        CompressedMatrix m(compressedMatrix.rows(),compressedMatrix.cols());
        m.setFromTriplets( incoming.begin(), incoming.end() );
        compressedMatrix += m;
        incoming.clear();
    }

    Index * getRowBegin() {
        compress();
        return compressedMatrix.outerIndexPtr();
    }

    Index * getColsIndex() {
        compress();
        return compressedMatrix.innerIndexPtr();
    }

    Real * getColsValue() {
        compress();
        return compressedMatrix.valuePtr();
    }


    /// Set all the entries of a row to 0
    void clearRow(Index i)
    {
        compress();
        for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
        {
            it.valueRef() = 0;
        }
    }

    /// Set all the entries of rows imin to imax-1 to 0.
    void clearRows(Index imin, Index imax)
    {
        compress();
        for(Index i=imin; i<imax; i++)
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
            {
                it.valueRef() = 0;
            }
    }

    ///< Set all the entries of a column to 0. Not efficient !
    void clearCol(Index col)
    {
        compress();
        for(Index i=0; i<compressedMatrix.rows(); i++ )
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
            {
                if( it.col()==col)
                {
                    it.valueRef() = 0;
                }
            }
    }

    ///< Clears the all the entries of column imin to column imax-1. Not efficient !
    void clearCols(Index imin, Index imax)
    {
        compress();
        for(Index i=0; i<compressedMatrix.rows(); i++ )
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it && it.col()<imax; ++it)
            {
                if( imin<=it.col() )
                    it.valueRef() = 0;
            }
    }

    ///< Set all the entries of column i and of row i to 0. Not efficient !
    void clearRowCol(Index i)
    {
        clearRow(i);
        clearCol(i);
    }

    ///< Clears all the entries of rows imin to imax-1 and columns imin to imax-1
    void clearRowsCols(Index imin, Index imax)
    {
        clearRows(imin,imax);
        clearCols(imin,imax);
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        Index r=rowSize(), c=colSize();
        resize(0,0);
        resize(r,c);
        incoming.clear();
    }

    /// Matrix-vector product
    void mult( VectorEigen& result, const VectorEigen& data )
    {
        compress();
        result = compressedMatrix * data;
    }

    /// Matrix-vector product openmp multithreaded
    void mult_MT( VectorEigen& result, const VectorEigen& data )
    {
        compress();
#ifdef _OPENMP
        result = linearsolver::mul_EigenSparseDenseMatrix_MT( compressedMatrix, data );
#else
        result = compressedMatrix * data;
#endif
    }

    /// Matrix-Vector product (dense vector with contiguous memory layout)
    template<class V1, class V2>
    void multVector( V1& output, const V2& input ){
        Eigen::Map<VectorEigen> mo(&output[0],output.size());
        Eigen::Map<const VectorEigen> mi(&input[0],input.size());
        compress();
        mo = compressedMatrix * mi;
    }

    /// Matrix-Vector product (dense vector with contiguous memory layout)
    template<class V>
    V operator* (const V& input){
        V output(this->rowSize());
        multVector(output,input);
        return output;
    }


    static const char* Name();

    // sparse solver support
protected:
    typedef Eigen::SimplicialCholesky<Eigen::SparseMatrix<Real> >  SimplicialCholesky;
    SimplicialCholesky cholesky; ///< used to factorize the matrix and solve systems using Cholesky method, for symmetric positive definite matrices only.
public:
    /// Try to compute the LDLT decomposition, and return true if success. The matrix is unchanged.
    bool choleskyDecompose()
    {
        compress();
        cholesky.compute(compressedMatrix);
        if( !cholesky.succeeded() )
        {
            msg_info()<<"EigenSparseSquareMatrix::factorize() failed" << std::endl;
            return false;
        }
        return true;
    }

    /// Solve Ax=b, where A is this matrix. WARNING: ldltDecompose() must be called first. x and b can be the same vector.
    void choleskySolve( VectorEigen& x, const VectorEigen& b ) const
    {
        x=b;
        // solve the equation
        cholesky.solveInPlace(x);
    }


    /// View this matrix as a MultiMatrix
    class MatrixAccessor: public core::behavior::MultiMatrixAccessor
    {
    public:

        MatrixAccessor( ThisMatrix* m=0 ) {setMatrix(m); }
        virtual ~MatrixAccessor() {}

        void setMatrix( ThisMatrix* m )
        {
            m->compress();
            matrix = m;
            matRef.matrix = m;
        }
        ThisMatrix* getMatrix() { return matrix; }
        const ThisMatrix* getMatrix() const { return matrix; }


        virtual int getGlobalDimension() const { return matrix->rowSize(); }
        virtual int getGlobalOffset(const core::behavior::BaseMechanicalState*) const { return 0; }
        virtual MatrixRef getMatrix(const core::behavior::BaseMechanicalState*) const
        {
            //    cerr<<"SingleMatrixAccessor::getMatrix" << endl;
            return matRef;
        }


        virtual InteractionMatrixRef getMatrix(const core::behavior::BaseMechanicalState* /*mstate1*/, const core::behavior::BaseMechanicalState* /*mstate2*/) const
        {
            assert(false);
            InteractionMatrixRef ref;
            return ref;
        }

    protected:
        ThisMatrix* matrix;   ///< The single matrix
        MatrixRef matRef; ///< The accessor to the single matrix

    };

    /// Get a view of this matrix as a MultiMatrix
    MatrixAccessor getAccessor() { return MatrixAccessor(this); }


    /// add this EigenBaseSparseMatrix to a BaseMatrix at the offset and multiplied by factor
    void addToBaseMatrix( BaseMatrix *matrix, SReal factor, Index offset ) const
    {
        for( Index j=0 ; j<compressedMatrix.outerSize() ; ++j )
            for( typename CompressedMatrix::InnerIterator it(compressedMatrix,j) ; it ; ++it )
            {
                matrix->add( offset+it.row(), offset+it.col(), factor*it.value() );
            }
    }


public:

    /// EigenBaseSparseMatrix multiplication
    /// res can be the same variable as this or rhs
    void mul(EigenBaseSparseMatrix<Real>& res, const EigenBaseSparseMatrix<Real>& rhs) const
    {
      ((EigenBaseSparseMatrix<Real>*)this)->compress();  /// \warning this violates the const-ness of the method
      ((EigenBaseSparseMatrix<Real>*)&rhs)->compress();  /// \warning this violates the const-ness of the parameter
      res.compressedMatrix = compressedMatrix * rhs.compressedMatrix;
    }

    /// EigenBaseSparseMatrix multiplication (openmp multithreaded version)
    /// @warning res MUST NOT be the same variable as this or rhs
    void mul_MT(EigenBaseSparseMatrix<Real>& res, const EigenBaseSparseMatrix<Real>& rhs) const
    {
    #ifdef _OPENMP
        assert( &res != this );
        assert( &res != &rhs );
        ((EigenBaseSparseMatrix<Real>*)this)->compress();  /// \warning this violates the const-ness of the method
        ((EigenBaseSparseMatrix<Real>*)&rhs)->compress();  /// \warning this violates the const-ness of the parameter
        conservative_sparse_sparse_product_selector_MT<CompressedMatrix,CompressedMatrix,CompressedMatrix>::run(compressedMatrix, rhs.compressedMatrix, res.compressedMatrix);
    #else
        mul( res, rhs );
    #endif
    }

    /// Sparse x Dense Matrix product
    void mul( Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>& res, const Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>& rhs )
    {
        res = compressedMatrix * rhs;
    }

    /// Sparse x Dense Matrix product openmp multithreaded
    void mul_MT( Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>& res, const Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>& rhs )
    {
        compress();
#ifdef _OPENMP
        res = linearsolver::mul_EigenSparseDenseMatrix_MT( compressedMatrix, rhs );
#else
        res = compressedMatrix * rhs;
#endif
    }


};

template<> inline const char* EigenBaseSparseMatrix<double>::Name() { return "EigenBaseSparseMatrixd"; }
template<> inline const char* EigenBaseSparseMatrix<float>::Name()  { return "EigenBaseSparseMatrixf"; }



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
