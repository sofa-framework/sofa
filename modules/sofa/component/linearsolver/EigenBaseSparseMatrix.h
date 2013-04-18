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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenBaseSparseMatrix_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenBaseSparseMatrix_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/SortedPermutation.h>
#include <sofa/helper/vector.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <map>
#include <Eigen/Sparse>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace linearsolver
{
using helper::vector;

//#define EigenBaseSparseMatrix_CHECK
//#define EigenBaseSparseMatrix_VERBOSE


/** Sparse matrix based on the Eigen library.

An Eigen::SparseMatrix<Real, RowMajor> matrix is used to store the data in Compressed Row Storage mode.
This matrix can not be accessed randomly. Two access modes are implemented.

The first access mode consists in inserting entries in increasing row, increasing column order.
Method beginRow(int index) must be called before any entry can be appended to row i.
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
    /** Impossible using this class
    */
    void set(int /*i*/, int /*j*/, double /*v*/)
    {
        assert( false && "EigenBaseSparseMatrix::set(int i, int j, double v) is not implemented !");
    }


public:

    typedef TReal Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef EigenBaseSparseMatrix<TReal> ThisMatrix;
    typedef Eigen::Triplet<Real> Triplet;

private:

    vector<Triplet> incoming;             ///< Scheduled additions


public:

    CompressedMatrix compressedMatrix;    ///< the compressed matrix


    EigenBaseSparseMatrix(int nbRow=0, int nbCol=0)
    {
        resize(nbRow,nbCol);
    }

    void setIdentity()
    {
        clear();
        for( unsigned i=0; i<rowSize(); i++ )
        {
            if( i==colSize() ) break;
            add(i,i,1.0);
        }
        compress();
    }

    /// Schedule the addition of the value at the given place. Scheduled additions must be finalized using function compress() .
    void add( int row, int col, SReal value ){
        incoming.push_back( Triplet(row,col,value) );
    }

    /// Insert in the compressed matrix. There must be no value at this place already. Efficient only if the value is inserted at the last place of the last row.
    void insertBack( int row, int col, Real value){
        compressedMatrix.insert(row,col) = value;
    }

    /// Return a reference to the given entry in the compressed matrix.There can (must ?) be a value at this place already. Efficient only if the it is at the last place of the compressed matrix.
    Real& coeffRef( int i, int j ){
        return compressedMatrix.coeffRef(i,j);
    }

    /** Clear and resize this to (m.rows,nbCol) and initialize it with the given matrix, columns shifted of the given value: this(i,j+shift) = m(i,j).
      @precond nbCol >= m.cols + shift
      */
    void copy(const EigenBaseSparseMatrix& m, unsigned nbCol, unsigned shift)
    {
        resize(m.rowSize(),nbCol);
        const CompressedMatrix& im = m.compressedMatrix;
        for(int i=0; i<im.rows(); i++)
        {
            for(typename CompressedMatrix::InnerIterator j(im,i); j; ++j)
                add(i,shift+j.col(),j.value());
        }
        compress();
    }


    /// Resize the matrix without preserving the data (the matrix is set to zero)
    void resize(int nbRow, int nbCol)
    {
        compressedMatrix.resize(nbRow,nbCol);
    }



    /// number of rows
    unsigned int rowSize(void) const
    {
        return compressedMatrix.rows();
    }

    /// number of columns
    unsigned int colSize(void) const
    {
        return compressedMatrix.cols();
    }

    SReal element(int i, int j) const
    {
        return compressedMatrix.coeff(i,j);
    }



    /// Add the values from the scheduled list, and clears the schedule list. @sa set(int i, int j, double v).
    void compress()
    {
        if( incoming.empty() ) return;
        CompressedMatrix m(compressedMatrix.rows(),compressedMatrix.cols());
        m.setFromTriplets( incoming.begin(), incoming.end() );
        compressedMatrix += m;
        incoming.clear();
    }


    /// Set all the entries of a row to 0, except the diagonal set to an extremely small number.
    void clearRow(int i)
    {
        compress();
        for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
        {
            if(it.index()==i) // diagonal entry
                it.valueRef()=(Real)1.0e-100;
            else it.valueRef() = 0;
        }
    }

    /// Set all the entries of rows imin to imax-1 to 0.
    void clearRows(int imin, int imax)
    {
        compress();
        for(int i=imin; i<imax; i++)
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
            {
                it.valueRef() = 0;
            }
    }

    ///< Set all the entries of a column to 0, except the diagonal set to an extremely small number.. Not efficient !
    void clearCol(int col)
    {
        compress();
        for(int i=0; i<compressedMatrix.rows(); i++ )
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it; ++it)
            {
                if( it.col()==col)
                {
                    if(it.index()==i) // diagonal entry
                        it.valueRef()=(Real)1.0e-100;
                    else it.valueRef() = 0;

                }
            }
    }

    ///< Clears the all the entries of column imin to column imax-1. Not efficient !
    void clearCols(int imin, int imax)
    {
        compress();
        for(int i=0; i<compressedMatrix.rows(); i++ )
            for (typename CompressedMatrix::InnerIterator it(compressedMatrix,i); it && it.col()<imax; ++it)
            {
                if( imin<=it.col() )
                    it.valueRef() = 0;
            }
    }

    ///< Set all the entries of column i and of row i to 0. Not efficient !
    void clearRowCol(int i)
    {
        clearRow(i);
        clearCol(i);
    }

    ///< Clears all the entries of rows imin to imax-1 and columns imin to imax-1
    void clearRowsCols(int imin, int imax)
    {
        clearRows(imin,imax);
        clearCols(imin,imax);
    }


    /// Set all values to 0, by resizing to the same size. @todo check that it really resets.
    void clear()
    {
        int r=rowSize(), c=colSize();
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
            std::cerr<<"EigenSparseSquareMatrix::factorize() failed" << std::endl;
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


        virtual InteractionMatrixRef getMatrix(const core::behavior::BaseMechanicalState* mstate1, const core::behavior::BaseMechanicalState* mstate2) const
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




};

template<> inline const char* EigenBaseSparseMatrix<double>::Name() { return "EigenBaseSparseMatrixd"; }
template<> inline const char* EigenBaseSparseMatrix<float>::Name()  { return "EigenBaseSparseMatrixf"; }



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
