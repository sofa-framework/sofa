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
//#include <Eigen/Core>
//#include <Eigen/SparseCholesky>
//#ifdef SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD
//#include <Eigen/CholmodSupport>
//#endif
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
public:

    typedef TReal Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;
    typedef EigenBaseSparseMatrix<TReal> ThisMatrix;

protected:
    // the auxiliary matrix used for random access
    typedef std::map<int,TReal> RowMap;   ///< Map which represents one row of the matrix. The index represents the column index of an entry.
    typedef std::map<int,RowMap> MatMap;  ///< Map which represents a matrix. The index represents the index of a row.
    MatMap incoming;                      ///< To store data before it is compressed in optimized format.
    CompressedMatrix compressedIncoming;            ///< auxiliary matrix to store the compressed version of the incoming matrix

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

    /** Clear and resize this to (m.rows,nbCol) and initialize it with the given matrix, columns shifted of the given value: this(i,j+shift) = m(i,j).
      @precond nbCol >= m.cols + shift
      */
    void copy(const EigenBaseSparseMatrix& m, unsigned nbCol, unsigned shift)
    {
        resize(m.rowSize(),nbCol);

        const CompressedMatrix& im = m.compressedMatrix;
        for(int i=0; i<im.rows(); i++)
        {
            compressedMatrix.startVec(i);
            for(typename CompressedMatrix::InnerIterator j(im,i); j; ++j)
                compressedMatrix.insertBack(i,shift+j.col())= j.value();
        }
        compressedMatrix.finalize();
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

    void add(int i, int j, double v)
    {
        if( v!=0.0 )
            incoming[i][j]+=(Real)v;
        //        cerr<<"EigenBaseSparseMatrix::set, size = "<< eigenMatrix.rows()<<", "<< eigenMatrix.cols()<<", entry: "<< i <<", "<<j<<" = "<< v << endl;
    }

    /// Converts the incoming matrix to compressedIncoming and clears the incoming matrix.
    void compress_incoming()
    {
        compressedIncoming.setZero();
        compressedIncoming.resize( compressedMatrix.rows(),compressedMatrix.cols() );
        if( incoming.empty() ) return;

        for( typename MatMap::const_iterator r=incoming.begin(),rend=incoming.end(); r!=rend; r++ )
        {
            int row = (*r).first;
            compressedIncoming.startVec(row);
            for( typename RowMap::const_iterator c=(*r).second.begin(),cend=(*r).second.end(); c!=cend; c++ )
            {
                int col = (*c).first;
                Real val = (*c).second;
                compressedIncoming.insertBack(row,col) = val;
            }
        }
        compressedIncoming.finalize();
        incoming.clear();
    }

    /// Add the values from the scheduled list, and clears the schedule list. @sa set(int i, int j, double v).
    void compress()
    {
        if( incoming.empty() ) return;
        compress_incoming();
        compressedMatrix += compressedIncoming;
    }

    /// must be called before inserting any element in the given row
    void beginRow( int i )
    {
        compressedMatrix.startVec(i);
    }

    /// This is efficient only if done in storing order: line, row
    void insertBack(int i, int j, double v)
    {
        if( v!=0.0 )
            compressedMatrix.insertBack(i,j) = (Real)v;
        //        cerr<<"EigenBaseSparseMatrix::set, size = "<< eigenMatrix.rows()<<", "<< eigenMatrix.cols()<<", entry: "<< i <<", "<<j<<" = "<< v << endl;
    }

    /** Schedule the replacement of the current value at row i and column j with value v. The replacement is effective only after method compress() is applied.
      If this method is used several times before compress() is applied, only the last value is used. @sa compress()
    */
    void set(int /*i*/, int /*j*/, double /*v*/)
    {
        cerr<<"EigenBaseSparseMatrix::set" << endl;
        assert( false && "EigenBaseSparseMatrix::set(int i, int j, double v) is not implemented !");
    }

//    /// Clears the matrix, sets the values from the scheduled list, and clears the replacement schedule list. @sa set(int i, int j, double v).
//    virtual void compressReplace()
//    {
//        if( incoming.empty() ) return;

//        Matrix cpy = eigenMatrix;
//        eigenMatrix.resize(eigenMatrix.rows(),eigenMatrix.cols());

//        typename MatMap::const_iterator r=incoming.begin(),rend=incoming.end();
//        for(int i=0; i<cpy.rows(); i++)
//        {
//            eigenMatrix.startVec(i);
//            while( r!=rend && (*r).first<i) r++; // find incoming values in the current row
//            if( r!=rend && (*r).first==i )
//            {
//                // there are incoming values in the current row, so interleave
//                typename Matrix::InnerIterator j(cpy,i);         // iterator on the previous matrix value
//                typename RowMap::const_iterator jj = (*r).second.begin(); // iterator on the incoming line
//                while( j && jj!=(*r).second.end() )
//                {
//                    if( j.col()<(*jj).first )   // value already present
//                    {
//                        eigenMatrix.insertBack(i,j.col())= j.value();
//                        ++j;
//                    }
//                    else    // incoming entry is inserted, or replace the current one
//                    {
//                        eigenMatrix.insertBack(i,(*jj).first) = (*jj).second;
//                        if(j.col()==(*jj).first) // replacement
//                            ++j;
//                        else
//                            jj++;
//                    }
//                }
//                // interleaving is over. One of the two lists may be not finished yet.
//                for(;j;++j)
//                    eigenMatrix.insertBack(i,j.col())= j.value();
//                for(;jj!=(*r).second.end();jj++)
//                    eigenMatrix.insertBack(i,(*jj).first) = (*jj).second;
//            }
//            else // no new values to insert, just copy the previous values
//            {
//                for(typename Matrix::InnerIterator j(cpy,i); j; ++j)
//                    eigenMatrix.insertBack(i,j.col())= j.value();
//            }
//        }
//        eigenMatrix.finalize();
//        incoming.clear();
//    }


//    void endEdit(){
////        compress();
//        compressedMatrix.finalize();
//    }

//    void clear(int i, int j)
//    {
//        compressedMatrix.coeffRef(i,j) = (Real)0;
//    }

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


    friend std::ostream& operator << (std::ostream& out, const EigenBaseSparseMatrix<TReal>& v )
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


//    /// Multiply the matrix by vector v and put the result in vector result
//    virtual void opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) const
//    {
//        result->resize(this->rowSize());
//        // map the vectors and perform the product on the maps
//        Eigen::Map<VectorEigen> vm( &((*v)[0]), v->size() );
//        Eigen::Map<VectorEigen> rm( &((*result)[0]), result->size() );
//        rm = eigenMatrix * vm;
//    }



};

template<> inline const char* EigenBaseSparseMatrix<double>::Name() { return "EigenBaseSparseMatrixd"; }
template<> inline const char* EigenBaseSparseMatrix<float>::Name()  { return "EigenBaseSparseMatrixf"; }



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
